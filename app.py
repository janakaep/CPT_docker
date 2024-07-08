import plotly.graph_objects as go
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, send_file, redirect, url_for, session, jsonify
import joblib
import os
from flask_session import Session

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load the model, scaler, polynomial features, and PCA
model = joblib.load('best_trained_model.pkl')
scaler = joblib.load('scaler3.pkl')
poly = joblib.load('poly3.pkl')
pca = joblib.load('pca3.pkl')

# Define the colors for the classes
colors = ['#FF0000', '#A52A2A', '#4682B4', '#0000FF', '#008080', '#2E8B57', '#9ACD32', '#FFFF00', '#FFA500', '#8B4513']
labels = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9', 'Class 10']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        output_folder = request.form.get('output_path')
        if not output_folder:
            return "Output path not provided"
        os.makedirs(output_folder, exist_ok=True)
        file_path = os.path.join(output_folder, file.filename)
        file.save(file_path)
        session['file_path'] = file_path
        session['output_folder'] = output_folder
        return redirect(url_for('process'))

@app.route('/process')
def process():
    file_path = session.get('file_path')
    output_folder = session.get('output_folder')
    if not file_path or not output_folder:
        return "Error: File path or output folder not found in session"

    new_data = pd.read_csv(file_path)
    data_non_null = new_data.dropna(subset=['SBT PP']).copy()
    data_null = new_data[new_data['SBT PP'].isnull()].copy()
    X = data_non_null[['Bq', 'Qt']].values
    X_scaled = scaler.transform(X)
    X_poly = poly.transform(X_scaled)
    X_pca = pca.transform(X_poly)
    y_pred = model.predict(X_pca)
    y_pred_classes = np.argmax(y_pred, axis=1)
    data_non_null.loc[:, 'Predicted SBT PP'] = y_pred_classes + 1
    data_combined = pd.concat([data_non_null, data_null], sort=False)
    output_path = os.path.join(output_folder, 'new_data_with_predictions.csv')
    data_combined.to_csv(output_path, index=False)

    # Create the Plotly scatter plot
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=url_for('static', filename='images/Robertson 1986log.jpg'),
            xref="x",
            yref="y",
            x=-0.2,
            y=np.log10(100),
            sizex=1.6,
            sizey=np.log10(100)-np.log10(0.1),
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            marker=dict(color=[colors[i] for i in y_pred_classes], showscale=False),
            text=y_pred_classes + 1,
            name='Data Points'
        )
    )
    for i, color in enumerate(colors):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                legendgroup='group',
                showlegend=True,
                name=labels[i]
            )
        )
    fig.update_layout(
        title='Predicted Scatter Plot',
        xaxis=dict(title='Bq', range=[-0.2, 1.4], scaleanchor='y', scaleratio=1),
        yaxis=dict(title='Qt', type='log', range=[np.log10(0.1), np.log10(100)], tickvals=[0.1, 1, 10, 100], ticktext=['0.1', '1', '10', '100']),
        width=1000,
        height=700,
        images=[dict(
            source=url_for('static', filename='images/Robertson 1986log.jpg'),
            xref="x",
            yref="y",
            x=-0.2,
            y=np.log10(100),
            sizex=1.6,
            sizey=np.log10(100)-np.log10(0.1),
            sizing="stretch",
            opacity=1,
            layer="below"
        )],
        legend=dict(title='Classes', itemsizing='constant')
    )
    plot_html_path = os.path.join(output_folder, 'scatter_plot.html')
    fig.write_html(plot_html_path, full_html=True)
    session['output_path'] = output_path
    session['plot_html_path'] = plot_html_path
    return redirect(url_for('results'))

@app.route('/results')
def results():
    output_path = session.get('output_path')
    plot_html_path = session.get('plot_html_path')
    if not output_path or not plot_html_path:
        return "Error: File not found"
    return render_template('results.html', plot_html_path=plot_html_path)

@app.route('/plot')
def plot():
    plot_html_path = session.get('plot_html_path')
    if not plot_html_path or not os.path.exists(plot_html_path):
        return "Error: Plot file not found"
    with open(plot_html_path, 'r') as f:
        plot_html = f.read()
    return plot_html

@app.route('/download_file')
def download_file():
    output_path = session.get('output_path')
    if not output_path or not os.path.exists(output_path):
        return "Error: File not found"
    return send_file(output_path, as_attachment=True)

@app.route('/update_classification', methods=['POST'])
def update_classification():
    data = request.json
    point_index = data.get('point_index')
    new_class = data.get('new_class')
    output_path = session.get('output_path')
    if not output_path:
        return jsonify({'status': 'error', 'message': 'Output path not found in session'})
    try:
        point_index = int(point_index)
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid point index'})
    new_data = pd.read_csv(output_path)
    if point_index is not None and 0 <= point_index < len(new_data):
        new_data.loc[point_index, 'Predicted SBT PP'] = new_class
    else:
        return jsonify({'status': 'error', 'message': 'Invalid point index'})
    new_data.to_csv(output_path, index=False)
    new_data['Predicted SBT PP'] = new_data['Predicted SBT PP'].astype(int)
    data_non_null = new_data.dropna(subset=['SBT PP']).copy()
    X = data_non_null[['Bq', 'Qt']].values
    y_pred_classes = data_non_null['Predicted SBT PP'].values - 1
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=url_for('static', filename='images/Robertson 1986log.jpg'),
            xref="x",
            yref="y",
            x=-0.2,
            y=np.log10(100),
            sizex=1.6,
            sizey=np.log10(100)-np.log10(0.1),
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            marker=dict(color=[colors[i] for i in y_pred_classes], showscale=False),
            text=y_pred_classes + 1,
            name='Data Points'
        )
    )
    for i, color in enumerate(colors):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                legendgroup='group',
                showlegend=True,
                name=labels[i]
            )
        )
    fig.update_layout(
        title='Predicted Scatter Plot',
        xaxis=dict(title='Bq', range=[-0.2, 1.4], scaleanchor='y', scaleratio=1),
        yaxis=dict(title='Qt', type='log', range=[np.log10(0.1), np.log10(100)], tickvals=[0.1, 1, 10, 100], ticktext=['0.1', '1', '10', '100']),
        width=1000,
        height=700,
        images=[dict(
            source=url_for('static', filename='images/Robertson 1986log.jpg'),
            xref="x",
            yref="y",
            x=-0.2,
            y=np.log10(100),
            sizex=1.6,
            sizey=np.log10(100)-np.log10(0.1),
            sizing="stretch",
            opacity=1,
            layer="below"
        )],
        legend=dict(title='Classes', itemsizing='constant')
    )
    plot_html_path = session.get('plot_html_path')
    fig.write_html(plot_html_path, full_html=True)
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)

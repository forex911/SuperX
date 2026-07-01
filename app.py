import os
import uuid
import numpy as np
import cv2
import requests
import onnxruntime as ort
from flask import Flask, request, render_template, send_from_directory, url_for, flash, redirect
from werkzeug.utils import secure_filename

# Make sure we're looking for templates in the right place
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = "super_secret_key_for_image_enhancement"

# Create necessary directories
app.config['UPLOAD_FOLDER'] = os.path.join(static_dir, 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(static_dir, 'results')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Ensure directories exist
os.makedirs(static_dir, exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# --- ONNX Model Setup ---
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'onnx_model'))
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, 'esrgan.onnx')
ONNX_DATA_PATH = os.path.join(MODEL_DIR, 'esrgan.onnx.data')

# HuggingFace model URLs
ONNX_MODEL_URL = "https://huggingface.co/SANJAI25/onnx_model/resolve/main/esrgan.onnx"
ONNX_DATA_URL = "https://huggingface.co/SANJAI25/onnx_model/resolve/main/esrgan.onnx.data"


def download_file(url, dest_path):
    """Download a file from URL to destination path with progress."""
    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"  Progress: {pct:.1f}% ({downloaded}/{total_size} bytes)", end='\r')
    print(f"\nDownloaded to: {dest_path} ({downloaded} bytes)")


def load_onnx_model():
    """Download ONNX model from HuggingFace if not cached, then load it."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Download model file if not present
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"ONNX model not found at {ONNX_MODEL_PATH}. Downloading from HuggingFace...")
        download_file(ONNX_MODEL_URL, ONNX_MODEL_PATH)
    else:
        print(f"ONNX model found at {ONNX_MODEL_PATH}")

    # Download data file if not present (contains the weights)
    if not os.path.exists(ONNX_DATA_PATH):
        print(f"ONNX data file not found at {ONNX_DATA_PATH}. Downloading from HuggingFace...")
        download_file(ONNX_DATA_URL, ONNX_DATA_PATH)
    else:
        print(f"ONNX data file found at {ONNX_DATA_PATH}")

    # Create ONNX Runtime session
    print("Loading ONNX Runtime session...")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Use CPU execution provider (works on all deployment platforms)
    session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        sess_options=session_options,
        providers=['CPUExecutionProvider']
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"ONNX model loaded. Input: '{input_name}', Output: '{output_name}'")
    return session, input_name, output_name


# Load the ONNX model at startup
session, input_name, output_name = load_onnx_model()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum file size is 50MB.')
    return redirect(url_for('index'))


@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/upload')
def index():
    return render_template('index.html')


@app.route('/enhance', methods=['POST'])
def enhance_image():
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            # Save the file with a unique filename
            original_filename = secure_filename(file.filename)
            base_filename = os.path.splitext(original_filename)[0]
            unique_id = str(uuid.uuid4())[:8]
            unique_filename = f"{base_filename}_{unique_id}"
            
            # Save the original file
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename + '.jpg')
            file.save(input_path)
            
            # Verify the file was saved correctly
            if not os.path.exists(input_path):
                flash(f'Error saving uploaded file to {input_path}')
                return redirect(url_for('index'))
                
            print(f"Image saved to: {input_path}")
            
            # Read and preprocess the image
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                flash('Error reading uploaded image. The file might be corrupted.')
                return redirect(url_for('index'))
            
            h, w = img.shape[:2]
            print(f"Original image size: {w}x{h}")
            
            # Automatically resize if image is too large
            max_input_dimension = 2000
            if max(h, w) > max_input_dimension:
                scale = max_input_dimension / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                print(f"Resizing from {w}x{h} to {new_w}x{new_h}")
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = new_h, new_w
                flash(f'Image automatically resized to {w}x{h} for processing. Output will be {w*4}x{h*4}.')
            
            print(f"Processing image size: {w}x{h}")
            
            # Process image in tiles to handle large images
            tile_size = 256  # Smaller tiles for better memory management
            overlap = 16
            scale = 4
            
            # Calculate output size
            output_h, output_w = h * scale, w * scale
            print(f"Output size will be: {output_w}x{output_h}")
            
            # Create output file path
            output_filename = unique_filename + '_enhanced.png'
            output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
            
            # Store tiles with their positions
            output_tiles = []
            
            # Process tiles
            tile_count = 0
            total_tiles = ((h + tile_size - overlap - 1) // (tile_size - overlap)) * ((w + tile_size - overlap - 1) // (tile_size - overlap))
            
            for i in range(0, h, tile_size - overlap):
                for j in range(0, w, tile_size - overlap):
                    # Extract tile
                    tile_h_start = i
                    tile_h_end = min(i + tile_size, h)
                    tile_w_start = j
                    tile_w_end = min(j + tile_size, w)
                    
                    tile = img[tile_h_start:tile_h_end, tile_w_start:tile_w_end, :]
                    
                    # Preprocess tile for ONNX: normalize and convert BGR->RGB, NCHW format
                    tile_normalized = tile.astype(np.float32) / 255.0
                    tile_input = np.transpose(tile_normalized[:, :, [2, 1, 0]], (2, 0, 1))  # BGR->RGB, HWC->CHW
                    tile_input = np.expand_dims(tile_input, axis=0)  # Add batch dimension: (1, 3, H, W)
                    
                    # Run ONNX inference
                    try:
                        tile_output = session.run(
                            [output_name],
                            {input_name: tile_input}
                        )[0]
                    except Exception as infer_error:
                        print(f"Inference error: {infer_error}")
                        flash('Error during image enhancement. Try a smaller image.')
                        return redirect(url_for('index'))
                    
                    # Post-process: squeeze batch dim, clip, convert back to uint8
                    tile_output = np.squeeze(tile_output, axis=0)  # Remove batch dim: (3, H, W)
                    tile_output = np.clip(tile_output, 0, 1)
                    tile_output = np.transpose(tile_output[[2, 1, 0], :, :], (1, 2, 0))  # RGB->BGR, CHW->HWC
                    tile_output = (tile_output * 255.0).round().astype(np.uint8)
                    
                    # Calculate output position
                    out_h_start = tile_h_start * scale
                    out_w_start = tile_w_start * scale
                    
                    # Store tile
                    output_tiles.append({
                        'data': tile_output,
                        'row': i // (tile_size - overlap),
                        'col': j // (tile_size - overlap),
                        'h_start': out_h_start,
                        'w_start': out_w_start
                    })
                    
                    tile_count += 1
                    print(f"Processed tile {tile_count}/{total_tiles}")
            
            print(f"All {tile_count} tiles processed. Stitching image...")
            
            # Group tiles by row
            rows = {}
            for tile_info in output_tiles:
                row_idx = tile_info['row']
                if row_idx not in rows:
                    rows[row_idx] = []
                rows[row_idx].append(tile_info)
            
            # Sort tiles in each row by column
            for row_idx in rows:
                rows[row_idx].sort(key=lambda x: x['col'])
            
            # Concatenate row by row
            row_images = []
            for row_idx in sorted(rows.keys()):
                row_tiles_data = [tile['data'] for tile in rows[row_idx]]
                row_img = np.concatenate(row_tiles_data, axis=1) if len(row_tiles_data) > 1 else row_tiles_data[0]
                row_images.append(row_img)
                
                # Clear memory
                del row_tiles_data
                print(f"Stitched row {row_idx + 1}/{len(rows)}")
            
            # Concatenate all rows
            print("Combining all rows...")
            output = np.concatenate(row_images, axis=0) if len(row_images) > 1 else row_images[0]
            
            print(f"Final image size: {output.shape[1]}x{output.shape[0]}")
            
            # Save the result
            output_filename = unique_filename + '_enhanced.png'
            output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
            success = cv2.imwrite(output_path, output)
            if not success:
                flash('Error saving enhanced image')
                return redirect(url_for('index'))
                
            print(f"Enhanced image saved to: {output_path}")
            
            # Generate static URLs for images
            original_filename_only = os.path.basename(input_path)
            original_static_url = f"/static/uploads/{original_filename_only}"
            enhanced_static_url = f"/static/results/{output_filename}"
            
            # Return the result page - try the simple template first
            try:
                # Try the simple result template first
                return render_template('simple_result.html', 
                                    original=original_filename_only,
                                    enhanced=output_filename,
                                    original_url=original_static_url,
                                    enhanced_url=enhanced_static_url)
            except Exception as simple_template_error:
                print(f"Error with simple template: {str(simple_template_error)}")
                try:
                    # Fallback to the original template
                    return render_template('result.html', 
                                        original=original_filename_only,
                                        enhanced=output_filename,
                                        original_url=original_static_url,
                                        enhanced_url=enhanced_static_url)
                except Exception as template_error:
                    flash(f'Error rendering templates: {str(template_error)}')
                    return redirect(url_for('index'))
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error details: {error_details}")
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a PNG, JPG, or JPEG file.')
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.abspath(app.config['UPLOAD_FOLDER']), filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(os.path.abspath(app.config['RESULTS_FOLDER']), filename)

@app.route('/direct-result/<filename>')
def direct_result(filename):
    """Direct route to display a specific enhanced image"""
    file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_from_directory(os.path.abspath(app.config['RESULTS_FOLDER']), filename)
    else:
        return f"File not found: {file_path}", 404

@app.route('/show-results')
def show_results():
    """Display all enhanced images"""
    results_path = os.path.abspath(app.config['RESULTS_FOLDER'])
    if not os.path.exists(results_path):
        return "Results directory not found", 404
        
    files = os.listdir(results_path)
    enhanced_images = [f for f in files if f.endswith('.png')]
    
    html = "<h1>Enhanced Images</h1>"
    html += "<ul>"
    for img in enhanced_images:
        img_url = url_for('result_file', filename=img)
        html += f'<li><img src="{img_url}" width="300"><br><a href="{img_url}" download>Download {img}</a></li>'
    html += "</ul>"
    
    return html

if __name__ == '__main__':
    app.run(debug=True)
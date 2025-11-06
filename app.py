import streamlit as st
import os
import glob
from PIL import Image
import numpy as np

# --- 0. TensorFlow Dependencies (Mandatory at the start) ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Activation, BatchNormalization, Dropout, Concatenate, Conv2DTranspose
except ImportError:
    st.error("❌ ERROR: TensorFlow is not installed. Ensure 'tensorflow-cpu==2.12.0' is in requirements.txt.")
    st.stop()
    
# ==============================================================================
# CONFIGURATION & MODEL ARCHITECTURE 
# ==============================================================================
IMG_SIZE = 256
# MODEL_G_PATH: Model is placed in the root folder for deployment stability
MODEL_G_PATH = 'pix2pix_tryon_G.h5' 

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential([
        Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)
    ])
    if apply_batchnorm: result.add(BatchNormalization())
    result.add(LeakyReLU(0.2))
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential([
        Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False),
        BatchNormalization()
    ])
    if apply_dropout: result.add(Dropout(0.5))
    result.add(Activation('relu'))
    return result

@st.cache_resource
def GeneratorUNet(input_shape=(IMG_SIZE, IMG_SIZE, 7), output_channels=3):
    # 5-layer UNet architecture (must match your 25MB model weights)
    inputs = Input(shape=input_shape)
    down_stack = [ downsample(32, 4, apply_batchnorm=False), downsample(64, 4), downsample(128, 4), downsample(256, 4), downsample(512, 4, apply_batchnorm=False) ]
    up_stack = [ upsample(256, 4, apply_dropout=True), upsample(128, 4), upsample(64, 4), upsample(32, 4) ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(output_channels, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')

    x = inputs; skips = []
    for down in down_stack: x = down(x); skips.append(x)
        
    x = up_stack[0](skips[-1]) 
    for up, skip in zip(up_stack[1:], reversed(skips[1:-1])): x = Concatenate()([x, skip]); x = up(x)
        
    x = Concatenate()([x, skips[0]])
    x = last(x)
    return Model(inputs=inputs, outputs=x, name='Generator')

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_assets(folder):
    # Checks for and retrieves asset file paths
    folder_path = os.path.join('assets', folder)
    if not os.path.exists(folder_path): 
        st.error(f"❌ KESALAHAN ASET: Folder '{folder_path}' not found.")
        st.stop()
    return sorted(glob.glob(os.path.join(folder_path, '*')))

def load_and_preprocess(img_data, is_mask=False):
    # Loads and resizes input image to 256x256
    img = Image.open(img_data) if isinstance(img_data, str) else Image.open(img_data)
    img = img.convert('L' if is_mask else 'RGB').resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img).astype(np.float32)
    if is_mask: img_np = img_np[..., np.newaxis]
    return (img_np / 127.5) - 1.0 # Normalizes to [-1, 1]

@st.cache_resource(show_spinner=False)
def load_generator_model(placeholder):
    # Function to load model weights only once
    placeholder.info("⏳ Memuat Model AI (Pix2Pix Generator 25MB). Ini hanya dilakukan sekali...")
    if not os.path.exists(MODEL_G_PATH):
        placeholder.error(f"❌ MODEL TIDAK DITEMUKAN: Harap letakkan file model '{MODEL_G_PATH}' di root folder.")
        return None
    try:
        netG = GeneratorUNet()
        netG.load_weights(MODEL_G_PATH)
        placeholder.success("🎉 Model AI siap digunakan!")
        return netG
    except Exception as e:
        placeholder.error(f"❌ Gagal memuat bobot model: Pastikan arsitektur UNet sudah benar. Error: {e}")
        return None

def process_inference(selected_shoe_path, input_feet_data, netG, col_result):
    # Function to run the Virtual Try-On prediction
    if netG is None: col_result.error("⚠️ Proses Try-On Gagal: Model AI tidak berhasil dimuat.") ; return

    with col_result:
        with st.spinner("Processing and generating the virtual try-on image..."):
            try:
                # Load and Preprocess Shoe (IC), Feet (IA)
                ic_img_np = load_and_preprocess(selected_shoe_path, is_mask=False) 
                ia_img_np = load_and_preprocess(input_feet_data, is_mask=False) 
                # SIMULATE MASK (IM) - Channel 1 (Filled with white 255)
                im_img_np = (np.full((IMG_SIZE, IMG_SIZE, 1), 255.0, dtype=np.float32) / 127.5) - 1.0 

                # Concatenate Input to (256, 256, 7) & Add Batch dimension
                input_tensor_7ch = np.concatenate([ia_img_np, ic_img_np, im_img_np], axis=-1)
                input_tensor_4d = np.expand_dims(input_tensor_7ch, axis=0) 

                # INFERENCE MODEL
                fake_image_tf = netG(input_tensor_4d, training=False)
                
                # Convert result back to [0, 1] for display
                fake_image_np = (fake_image_tf.numpy()[0] * 0.5) + 0.5
                fake_image_display = np.clip(fake_image_np, 0, 1)

                st.subheader("Hasil Virtual Try-On")
                st.image(fake_image_display, caption="Generated New Shoe Result", use_column_width=True)

            except Exception as e:
                st.error(f"An error occurred during Inference: {e}")


# ==============================================================================
# STREAMLIT UI
# ==============================================================================

st.set_page_config(layout="wide", page_title="👟 Virtual Try-On Sepatu GAN")
st.title("👟 Virtual Try-On Sepatu")

# 1. Load Model with Dynamic Notification
model_status_placeholder = st.empty() 
netG = load_generator_model(model_status_placeholder)

# 2. Check Assets
shoe_assets = get_assets('shoes')
feet_assets = get_assets('feet')
if not shoe_assets or not feet_assets:
    st.error("❌ **ASSET ERROR:** Ensure 'assets/shoes/' and 'assets/feet/' folders contain images.")
    st.stop() 

# Initialize Session State
if 'selected_shoe_path' not in st.session_state: st.session_state['selected_shoe_path'] = None
    
# --- CATALOG DISPLAY ---
st.header("1. Pilih Sepatu (IC)")
st.markdown("---")

cols = st.columns(len(shoe_assets))

for i, path in enumerate(shoe_assets):
    shoe_name = os.path.basename(path)
    is_selected = st.session_state['selected_shoe_path'] == path

    with cols[i]:
        st.image(path, caption=shoe_name, use_column_width=True)
        
        button_label = f"✅ Dipilih" if is_selected else "Pilih Sepatu ini"
        button_type = "primary" if is_selected else "secondary" 
        
        if st.button(button_label, key=f"select_shoe_{i}", use_container_width=True, type=button_type):
            st.session_state['selected_shoe_path'] = path
            st.session_state['selected_shoe_name'] = shoe_name
            st.rerun() 

# --- FEET INPUT & TRY-ON MENU ---
if st.session_state['selected_shoe_path']:
    
    st.markdown("---")
    st.subheader(f"Sepatu Dipilih: {st.session_state['selected_shoe_name']}")
    st.header("2. Input Kaki (IA) & Try-On")
    
    col_input, col_result = st.columns([1, 2])
    
    with col_input:
        
        input_method = st.radio("Pilih Metode Input Kaki:", ("Pilih dari Galeri", "Unggah Citra Baru"), key='input_method_radio')
        input_feet_data = None 
        
        if input_method == "Pilih dari Galeri":
            feet_options = [os.path.basename(p) for p in feet_assets]
            selected_feet_name = st.selectbox("Pilih Bentuk Kaki:", feet_options, index=0, key='select_feet_gallery')
            input_feet_data = os.path.join('assets', 'feet', selected_feet_name)
            
        else:
            uploaded_file = st.file_uploader("Unggah Citra Kaki (JPG, PNG)", type=["jpg", "png", "jpeg"], key='feet_uploader')
            if uploaded_file is not None: input_feet_data = uploaded_file
        
        if input_feet_data is not None and netG is not None:
            st.image(input_feet_data, caption="Pratinjau Kaki Pilihan", width=200)

        st.markdown("<br>", unsafe_allow_html=True)
        tryon_disabled = netG is None or input_feet_data is None
        
        if st.button("✨ Terapkan Virtual Try-On", key='tryon_button', use_container_width=True, disabled=tryon_disabled):
            process_inference(st.session_state['selected_shoe_path'], input_feet_data, netG, col_result)
        
        if tryon_disabled and netG is None:
            st.warning("⚠️ Try-On dinonaktifkan: Model AI gagal dimuat.")
        elif tryon_disabled and input_feet_data is None:
            st.warning("Mohon pilih atau unggah citra kaki terlebih dahulu.")
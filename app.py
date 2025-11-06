import streamlit as st
import os
import glob
from PIL import Image
import numpy as np

# --- 0. Dependensi TensorFlow (Wajib di awal) ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Activation, BatchNormalization, Dropout, Concatenate, Conv2DTranspose
except ImportError:
    st.error("❌ ERROR: TensorFlow tidak terinstal. Pastikan 'tensorflow-cpu' ada di requirements.txt.")
    st.stop()
    
# ==============================================================================
# KONFIGURASI & MODEL ARCHITECTURE (Dipertahankan sama persis dengan bobot Anda)
# ==============================================================================
IMG_SIZE = 256
MODEL_G_PATH = 'models/pix2pix_tryon_G.h5' 

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
    if not os.path.exists('assets'): os.makedirs('assets')
    folder_path = os.path.join('assets', folder)
    if not os.path.exists(folder_path): os.makedirs(folder_path) 
    return sorted(glob.glob(os.path.join(folder_path, '*')))

def load_and_preprocess(img_data, is_mask=False):
    img = Image.open(img_data) if isinstance(img_data, str) else Image.open(img_data)
    img = img.convert('L' if is_mask else 'RGB').resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img).astype(np.float32)
    if is_mask: img_np = img_np[..., np.newaxis]
    return (img_np / 127.5) - 1.0 

@st.cache_resource(show_spinner=False)
def load_generator_model(placeholder):
    placeholder.info("⏳ Memuat Model AI (Pix2Pix Generator). Ini hanya dilakukan sekali...")
    if not os.path.exists(MODEL_G_PATH):
        placeholder.error(f"❌ MODEL TIDAK DITEMUKAN: Harap letakkan file model '{MODEL_G_PATH}'.")
        return None
    try:
        netG = GeneratorUNet()
        netG.load_weights(MODEL_G_PATH)
        placeholder.success("🎉 Model AI siap digunakan!") # Notif sukses
        # Placeholder akan tetap tampil sampai ada yang menggantikannya
        return netG
    except Exception as e:
        placeholder.error(f"❌ Gagal memuat bobot model: File model mungkin rusak. Error: {e}")
        return None

def process_inference(selected_shoe_path, input_feet_data, netG, col_result):
    if netG is None: col_result.error("⚠️ Proses Try-On Gagal: Model AI tidak berhasil dimuat.") ; return

    with col_result:
        with st.spinner("Sedang memproses dan menghasilkan citra virtual try-on..."):
            try:
                # Muat dan Preprocessing
                ic_img_np = load_and_preprocess(selected_shoe_path, is_mask=False) 
                ia_img_np = load_and_preprocess(input_feet_data, is_mask=False) 
                im_img_np = (np.full((IMG_SIZE, IMG_SIZE, 1), 255.0, dtype=np.float32) / 127.5) - 1.0 # Masker Simulasi

                # Gabungkan Input
                input_tensor_7ch = np.concatenate([ia_img_np, ic_img_np, im_img_np], axis=-1)
                input_tensor_4d = np.expand_dims(input_tensor_7ch, axis=0) 

                # INFERENCE & Output
                fake_image_tf = netG(input_tensor_4d, training=False)
                fake_image_np = (fake_image_tf.numpy()[0] * 0.5) + 0.5
                fake_image_display = np.clip(fake_image_np, 0, 1)

                st.subheader("Hasil Virtual Try-On")
                st.image(fake_image_display, caption="Hasil Generasi Sepatu Baru", use_column_width=True)

            except Exception as e:
                st.error(f"Terjadi Kesalahan saat Inference: {e}")

# ==============================================================================
# STREAMLIT UI
# ==============================================================================

st.set_page_config(layout="wide", page_title="Virtual Try-On Sepatu GAN")
st.title("👟 Virtual Try-On Sepatu")

# 1. Muat Model dengan Notifikasi Dinamis
model_status_placeholder = st.empty() 
netG = load_generator_model(model_status_placeholder)

# 2. Periksa Aset
shoe_assets = get_assets('shoes')
feet_assets = get_assets('feet')
if not shoe_assets or not feet_assets:
    st.error("❌ **KESALAHAN ASET:** Pastikan folder `assets/shoes/` dan `assets/feet/` terisi gambar.")
    st.stop() 

# Inisialisasi Session State
if 'selected_shoe_path' not in st.session_state: st.session_state['selected_shoe_path'] = None
if 'selected_shoe_name' not in st.session_state: st.session_state['selected_shoe_name'] = None
    
# --- TAMPILAN KATALOG (Klik pada Gambar) ---
st.header("1. Pilih Sepatu (IC)")
st.markdown("---")

cols = st.columns(len(shoe_assets))

for i, path in enumerate(shoe_assets):
    shoe_name = os.path.basename(path)
    is_selected = st.session_state['selected_shoe_path'] == path

    with cols[i]:
        # Tampilkan Gambar
        st.image(path, caption=shoe_name, use_column_width=True)
        
        # Tombol untuk memilih (diletakkan di bawah gambar, tetapi tombolnya sendiri kecil/tersembunyi)
        # Kita membuat tombol yang menarik perhatian
        button_label = f"✅ Dipilih" if is_selected else "Pilih Sepatu ini"
        button_type = "secondary" if not is_selected else "primary"
        
        if st.button(button_label, key=f"select_shoe_{i}", use_container_width=True, type=button_type):
            st.session_state['selected_shoe_path'] = path
            st.session_state['selected_shoe_name'] = shoe_name
            st.rerun() 

# --- MENU INPUT KAKI & TRY-ON ---
if st.session_state['selected_shoe_path']:
    
    st.markdown("---")
    st.header(f"Sepatu Dipilih: {st.session_state['selected_shoe_name']}")
    st.subheader("2. Input Kaki (IA) & Try-On")
    
    # Membagi layout untuk Input dan Hasil
    col_input, col_result = st.columns([1, 2])
    
    with col_input:
        
        input_method = st.radio("Pilih Metode Input Kaki:", ("Pilih dari Galeri", "Unggah Citra Baru"), key='input_method_radio')
        input_feet_data = None 
        
        # OPSI GALERI
        if input_method == "Pilih dari Galeri":
            feet_options = [os.path.basename(p) for p in feet_assets]
            selected_feet_name = st.selectbox("Pilih Bentuk Kaki:", feet_options, index=0, key='select_feet_gallery')
            input_feet_data = os.path.join('assets', 'feet', selected_feet_name)
            
        # OPSI UNGGAH
        else:
            uploaded_file = st.file_uploader("Unggah Citra Kaki (JPG, PNG)", type=["jpg", "png", "jpeg"], key='feet_uploader')
            if uploaded_file is not None: input_feet_data = uploaded_file
        
        # Menampilkan Pratinjau Kaki
        if input_feet_data is not None:
            st.image(input_feet_data, caption="Pratinjau Kaki Pilihan", width=200)

        # TOMBOL TRY-ON
        st.markdown("<br>", unsafe_allow_html=True)
        tryon_disabled = netG is None or input_feet_data is None
        
        if st.button("✨ Terapkan Virtual Try-On", key='tryon_button', use_container_width=True, disabled=tryon_disabled):
            process_inference(st.session_state['selected_shoe_path'], input_feet_data, netG, col_result)
        
        if tryon_disabled and netG is None:
            st.warning("⚠️ Try-On dinonaktifkan karena Model AI gagal dimuat. Periksa folder `models/`.")
        elif tryon_disabled and input_feet_data is None:
            st.warning("Mohon pilih atau unggah citra kaki terlebih dahulu.")
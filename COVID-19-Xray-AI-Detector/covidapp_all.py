import os
import io
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

st.set_page_config(
    page_title="COVID-19 X-ray AI Detector",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff, #f8fafc, #e0f2fe);
}
.title {
    text-align:center;
    font-size:42px;
    font-weight:900;
    color:#0f172a;
}
.subtitle {
    text-align:center;
    color:#475569;
    font-size:17px;
}
.name {
    text-align:center;
    color:#2563eb;
    font-size:18px;
    font-weight:700;
    margin-bottom:20px;
}
.warning {
    background:#fff7ed;
    padding:15px;
    border-radius:15px;
    border-left:5px solid #f97316;
    color:#7c2d12;
    margin-bottom:20px;
}
.explain-box {
    background:#f8fafc;
    padding:18px;
    border-radius:15px;
    border-left:5px solid #2563eb;
    margin-top:15px;
}
.footer {
    text-align:center;
    color:#64748b;
    margin-top:30px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ai_model():
    model = tf.keras.models.load_model("final_model.h5", compile=False)
    return model

model = load_ai_model()

class_names = ["COVID", "Normal", "Viral Pneumonia"]
last_conv_layer_name = "out_relu"

def get_explanation(predicted_class):
    if predicted_class == "COVID":
        return {
            "title": "COVID-19 Infection Pattern",
            "effect": "COVID-19 may affect the lungs by causing inflammation in lung tissues and reducing normal air exchange.",
            "cause": "It is commonly caused by SARS-CoV-2 virus infection.",
            "xray": "In X-ray images, COVID-related changes may appear as cloudy or patchy lung regions.",
            "advice": "This is only an AI prediction. Doctor consultation and clinical tests are required for confirmation."
        }

    elif predicted_class == "Viral Pneumonia":
        return {
            "title": "Viral Pneumonia Pattern",
            "effect": "Viral pneumonia may affect the lungs by causing infection and inflammation in air sacs.",
            "cause": "It can be caused by respiratory viruses that infect the lungs.",
            "xray": "In X-ray images, pneumonia may show patchy opacities or abnormal lung shadows.",
            "advice": "This result should be confirmed by a healthcare professional."
        }

    else:
        return {
            "title": "Normal Chest X-ray Pattern",
            "effect": "The AI model did not detect strong COVID or pneumonia-like abnormal patterns.",
            "cause": "This suggests the uploaded X-ray appears closer to normal class based on training data.",
            "xray": "Normal X-rays usually show clear lung fields without major abnormal cloudy regions.",
            "advice": "If symptoms are present, consult a doctor even if AI prediction is normal."
        }

def predict_image(img):
    img_rgb = img.convert("RGB")
    img_resized = img_rgb.resize((128, 128))

    img_array = np.array(img_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction) * 100)

    return predicted_class, confidence, prediction, img_array

def make_gradcam(img, img_array):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)

    if max_value == 0:
        return img.convert("RGB")

    heatmap = heatmap / max_value
    heatmap = heatmap.numpy()

    original = img.convert("RGB")

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize(original.size)
    heatmap_img = heatmap_img.convert("L")

    red_overlay = Image.new("RGB", original.size, (255, 0, 0))
    overlay = Image.composite(red_overlay, original, heatmap_img)
    overlay = Image.blend(original, overlay, alpha=0.35)

    return overlay

def create_pdf_report(predicted_class, confidence, prediction, explanation):
    buffer = io.BytesIO()

    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, height - 60, "COVID-19 X-ray AI Detection Report")

    pdf.setFont("Helvetica", 12)
    pdf.drawString(50, height - 105, f"Prediction: {predicted_class}")
    pdf.drawString(50, height - 130, f"Confidence: {confidence:.2f}%")

    pdf.drawString(50, height - 170, "Class Probabilities:")

    y = height - 195
    for i, name in enumerate(class_names):
        percent = float(prediction[0][i] * 100)
        pdf.drawString(70, y, f"{name}: {percent:.2f}%")
        y -= 25

    y -= 20
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(50, y, explanation["title"])

    pdf.setFont("Helvetica", 11)
    y -= 25
    pdf.drawString(50, y, f"Effect: {explanation['effect'][:90]}")
    y -= 20
    pdf.drawString(50, y, f"Cause: {explanation['cause'][:90]}")
    y -= 20
    pdf.drawString(50, y, f"X-ray Finding: {explanation['xray'][:90]}")
    y -= 20
    pdf.drawString(50, y, f"Note: {explanation['advice'][:90]}")

    pdf.setFont("Helvetica-Oblique", 10)
    pdf.drawString(
        50,
        80,
        "Disclaimer: This report is for educational/project purpose only. Not for real medical diagnosis."
    )

    pdf.drawString(
        50,
        55,
        "Developed by Priyadharshini | TensorFlow • CNN • Grad-CAM • Streamlit"
    )

    pdf.save()
    buffer.seek(0)

    return buffer

st.markdown('<div class="title">🩺 COVID-19 X-ray AI Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Prediction + Grad-CAM + Explanation + PDF Report</div>', unsafe_allow_html=True)
st.markdown('<div class="name">👩‍💻 Developed by Priyadharshini</div>', unsafe_allow_html=True)

st.markdown("""
<div class="warning">
⚠️ This app is for educational/project purpose only. It is not a real medical diagnosis tool.
</div>
""", unsafe_allow_html=True)

st.sidebar.title("📌 Options")

app_mode = st.sidebar.radio(
    "Choose App Mode",
    ["Normal Prediction", "Prediction + Grad-CAM"]
)

image_mode = st.sidebar.radio(
    "Choose Image Source",
    ["Upload X-ray Image", "Use Sample Dataset Image"]
)

st.sidebar.markdown("---")
st.sidebar.success("Loaded Model Type: Keras")
st.sidebar.info("Grad-CAM Layer: out_relu")
st.sidebar.info("Input Size: 128 × 128")

selected_img = None
caption = ""

if image_mode == "Upload X-ray Image":

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        selected_img = Image.open(uploaded_file)
        caption = "Uploaded X-ray"

else:

    category = st.selectbox(
        "Choose Category",
        ["Covid", "Normal", "Viral Pneumonia"]
    )

    sample_path = os.path.join(
        "archive",
        "Covid19-dataset",
        "test",
        category
    )

    if os.path.exists(sample_path):

        image_files = [
            f for f in os.listdir(sample_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(image_files) > 0:
            selected_file = st.selectbox("Choose X-ray Image", image_files)

            selected_img = Image.open(
                os.path.join(sample_path, selected_file)
            )

            caption = f"{category} / {selected_file}"

        else:
            st.error("No images found in this folder.")

    else:
        st.error("Dataset folder not found.")

if selected_img is not None:

    predicted_class, confidence, prediction, img_array = predict_image(selected_img)
    explanation = get_explanation(predicted_class)

    if app_mode == "Prediction + Grad-CAM":

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original X-ray")
            st.image(
                selected_img,
                caption=caption,
                use_column_width=True
            )

        with col2:
            st.subheader("🔥 Grad-CAM Overlay")
            gradcam_img = make_gradcam(selected_img, img_array)

            st.image(
                gradcam_img,
                caption="AI Focus Area",
                use_column_width=True
            )

    else:

        st.subheader("📷 X-ray Image")
        st.image(
            selected_img,
            caption=caption,
            use_column_width=True
        )

    st.markdown("---")

    st.subheader("🤖 AI Prediction Result")

    if predicted_class == "COVID":
        st.error(f"⚠️ Prediction: {predicted_class}")

    elif predicted_class == "Normal":
        st.success(f"✅ Prediction: {predicted_class}")

    else:
        st.warning(f"🟡 Prediction: {predicted_class}")

    st.info(f"Confidence: {confidence:.2f}%")
    st.progress(confidence / 100)

    st.write("### 📊 Class Probabilities")

    for i, name in enumerate(class_names):
        percent = float(prediction[0][i] * 100)
        st.write(f"{name}: {percent:.2f}%")
        st.progress(percent / 100)

    st.write("### 🧾 AI Explanation")

    st.markdown(f"""
    <div class="explain-box">
    <b>{explanation["title"]}</b><br><br>
    <b>How it may affect lungs:</b> {explanation["effect"]}<br><br>
    <b>Possible reason:</b> {explanation["cause"]}<br><br>
    <b>X-ray pattern:</b> {explanation["xray"]}<br><br>
    <b>Important note:</b> {explanation["advice"]}
    </div>
    """, unsafe_allow_html=True)

    pdf_file = create_pdf_report(
        predicted_class,
        confidence,
        prediction,
        explanation
    )

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_file,
        file_name="covid_xray_report.pdf",
        mime="application/pdf"
    )

else:
    st.info("Please upload or choose a sample X-ray image.")

st.markdown("""
<div class="footer">
🚀 Developed by Priyadharshini <br>
Built using TensorFlow • CNN • Grad-CAM • Streamlit • PDF Report
</div>
""", unsafe_allow_html=True)

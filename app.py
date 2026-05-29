import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image

# ---------------------------
# Roboflow API
# ---------------------------

rf = Roboflow(api_key="q17RjoLPdYRrtjjiDSe1")

st.write("Loading Weld Model...")

# Weld bead model
weld_project = rf.workspace("ishita-banerjee-s-workspace").project("weld-defect-rfhza")
weld_model = weld_project.version(4).model

st.write("Loading Defect Model...")

# Defect model
defect_project = rf.workspace("ishita-banerjee-s-workspace").project("weld-defects-final")
defect_model = defect_project.version(4).model

st.write("Models Loaded Successfully ✅")


# ---------------------------
# Colors
# ---------------------------

colors = {
    "crack": (255, 0, 0),          # Blue
    "porosity": (0, 255, 0),       # Green
    "spatter": (0, 0, 255),        # Red
    "burning": (255, 255, 0),      # Cyan
    "poor_forming": (255, 0, 255)  # Magenta
}


# ---------------------------
# Streamlit UI
# ---------------------------

st.title("Weld Defect Detection")

uploaded_file = st.file_uploader(
    "Upload Weld Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    # ---------------------------
    # Read Image
    # ---------------------------

    image = Image.open(uploaded_file)
    image = np.array(image)

    original_image = image.copy()

    # Save temp image
    temp_path = "temp.jpg"

    cv2.imwrite(
        temp_path,
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )

    # ---------------------------
    # STEP 1 : Detect Weld Bead
    # ---------------------------

    weld_pred = weld_model.predict(
        temp_path,
        confidence=35
    ).json()

    if len(weld_pred["predictions"]) > 0:

        for weld in weld_pred["predictions"]:

            x = int(weld["x"])
            y = int(weld["y"])
            w = int(weld["width"])
            h = int(weld["height"])

            x1 = max(0, int(x - w/2))
            y1 = max(0, int(y - h/2))

            x2 = min(image.shape[1], int(x + w/2))
            y2 = min(image.shape[0], int(y + h/2))

            # ---------------------------
            # Crop Weld Bead
            # ---------------------------

            weld_crop = image[y1:y2, x1:x2]

            # Save crop
            cv2.imwrite(
                "weld_crop.jpg",
                cv2.cvtColor(weld_crop, cv2.COLOR_RGB2BGR)
            )

            # ---------------------------
            # STEP 2 : Detect Defects
            # ---------------------------

            defect_pred = defect_model.predict(
                "weld_crop.jpg",
                confidence=5,
                overlap=30
            ).json()

            # ---------------------------
            # Draw Defect Boxes
            # ---------------------------

            for pred in defect_pred["predictions"]:

                px = int(pred["x"])
                py = int(pred["y"])
                pw = int(pred["width"])
                ph = int(pred["height"])

                label = pred["class"]
                conf = pred["confidence"]

                # Convert crop coords to original image coords
                x1_box = int(x1 + (px - pw/2))
                y1_box = int(y1 + (py - ph/2))

                x2_box = int(x1 + (px + pw/2))
                y2_box = int(y1 + (py + ph/2))

                # Class color
                color = colors.get(label, (255,255,255))

                # Draw rectangle
                cv2.rectangle(
                    image,
                    (x1_box, y1_box),
                    (x2_box, y2_box),
                    color,
                    3
                )

                # Label text
                text = f"{label} {conf:.2f}"

                cv2.putText(
                    image,
                    text,
                    (x1_box, y1_box - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    3
                )

            # ---------------------------
            # Draw Weld Bead Boundary
            # ---------------------------

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255,255,255),
                2
            )

    else:
        st.write("No weld bead detected")

    # ---------------------------
    # Show Final Output
    # ---------------------------

    st.image(
        image,
        caption="Detected Defects",
        use_container_width=True
    )
import gradio as gr
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('modeldementia.h5')

def predict_dementia(visit, mr_delay, m_f, hand, age, educ, ses, mmse, cdr, etiv, nwbv, asf, age_group):
    # Prepare the input features as expected by the model
    # The order of features and preprocessing should match the model's training process
    
    # One-hot encoding for 'M/F', 'Hand', and 'AgeGroup'
    # Adjust these encodings to match the training process
    m_f_encoded = [1 if m_f == 'M' else 0, 1 if m_f == 'F' else 0]
    hand_encoded = [1 if hand == 'R' else 0, 1 if hand == 'L' else 0]
    age_group_encoded = [1 if age_group == grp else 0 for grp in ['60-69', '70-79', '80-89', '90-99']]
    
    input_features = [visit, mr_delay, age, educ, ses, mmse, cdr, etiv, nwbv, asf] + \
                     [1 if m_f == 'M' else 0, 1 if m_f == 'F' else 0] + \
                     [1 if hand == 'R' else 0, 1 if hand == 'L' else 0] + \
                     [1 if age_group == grp else 0 for grp in ['70-79', '80-89', '90-99']]
    input_data = np.array(input_features).reshape(1, -1)
    
    # Make a prediction
    prediction = model.predict(input_data)
    predicted_label = 'Nondemented' if prediction[0] < 0.5 else 'Demented'
    return predicted_label

# Define the input fields for the Gradio interface
input_fields = [
    gr.Number(label="Visit"),
    gr.Number(label="MR Delay"),
    gr.Radio(["M", "F"], label="Gender"),
    gr.Radio(["R", "L"], label="Hand"),
    gr.Number(label="Age"),
    gr.Number(label="EDUC"),
    gr.Number(label="SES"),
    gr.Number(label="MMSE"),
    gr.Number(label="CDR"),
    gr.Number(label="eTIV"),
    gr.Number(label="nWBV"),
    gr.Number(label="ASF"),
    gr.Dropdown(['70-79', '80-89', '90-99'], label="Age Group")
]

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_dementia,
    inputs=input_fields,
    outputs="text",
    title="Dementia Prediction Model",
    description="Input the features to predict dementia status"
)

# Launch the Gradio app
iface.launch()

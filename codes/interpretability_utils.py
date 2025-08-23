import numpy as np
import shap
# from tensorflow.keras.models import load_model
from positional_encoding import PositionalEncoding
from data_preprocessing_utils import load_K562_encoded_by_both_base_and_base_pair

def compute_shap_values(model):
  
    # Computes SHAP values using training data, with KernelExplainer.

    samples_size=50
    background_sample_size = 500
    num_clusters = 200
    perturbations_per_sample=1000

    np.random.seed(42)

    kX, kXont, kXofft, _, kon_epi, koff_epi = load_K562_encoded_by_both_base_and_base_pair()
    epi_rows, epi_cols = kon_epi.shape[1], kon_epi.shape[2]
    epi_dim = epi_rows * epi_cols

    concat_data = np.concatenate([
        kX, kXont, kXofft,
        kon_epi.reshape(len(kon_epi), -1),
        koff_epi.reshape(len(koff_epi), -1)
    ], axis=1)

    x_data_indices = np.random.choice(len(concat_data), size=background_sample_size, replace=False)
    x_summary = shap.kmeans(concat_data[x_data_indices], num_clusters)

    def shap_predictions(data):
        feat_dim, ont_dim, offt_dim = kX.shape[1], kXont.shape[1], kXofft.shape[1]
        epi_start = feat_dim + ont_dim + offt_dim
        input_1 = data[:, :feat_dim]
        input_2 = data[:, feat_dim:feat_dim+ont_dim]
        input_3 = data[:, feat_dim+ont_dim:feat_dim+ont_dim+offt_dim]
        input_4 = data[:, epi_start:epi_start+epi_dim].reshape((-1, epi_rows, epi_cols))
        input_5 = data[:, epi_start+epi_dim:epi_start+2*epi_dim].reshape((-1, epi_rows, epi_cols))
        return model.predict([input_1, input_2, input_3, input_4, input_5])

    data_indices = np.random.choice(len(concat_data), size=samples_size, replace=False)
    random_data = concat_data[data_indices]

    # Compute SHAP values with KernelExplainer
    shap_explainer = shap.KernelExplainer(shap_predictions, x_summary)
    shap_values = shap_explainer.shap_values(random_data, nsamples=perturbations_per_sample)

    np.save("shap_values.npy", shap_values)

# if __name__ == "__main__":
#     model = load_model("tcrispr_model.h5", custom_objects={"PositionalEncoding": PositionalEncoding})
#     compute_shap_values(model)
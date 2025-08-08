import numpy as np
import shap
from sklearn.cluster import KMeans


def compute_shap_values(trainer, model):
  
    # Computes SHAP values using training data, with KernelExplainer.

    samples_size=50
    background_sample_size = 500
    num_clusters = 200
    perturbations_per_sample=1000

    np.random.seed(42)

    concat_train_data = np.concatenate([
        trainer.train_features, trainer.train_feature_ont, trainer.train_feature_offt,
        trainer.train_on_epigenetic_code.reshape(len(trainer.train_on_epigenetic_code), -1),
        trainer.train_off_epigenetic_code.reshape(len(trainer.train_off_epigenetic_code), -1),], axis=1)

    train_data_indices = np.random.choice(len(concat_train_data), size=background_sample_size, replace=False)
    random_train_data = concat_train_data[train_data_indices]

    train_summary = shap.kmeans(random_train_data, num_clusters)

    def shap_predictions(data):
        
        feat_dim = trainer.train_features.shape[1]
        ont_dim = trainer.train_feature_ont.shape[1]
        offt_dim = trainer.train_feature_offt.shape[1]
        epi_dim = 24 * 4 
        
        input_1 = data[:, :feat_dim]
        input_2 = data[:, feat_dim:feat_dim+ont_dim]
        input_3 = data[:, feat_dim+ont_dim:feat_dim+ont_dim+offt_dim]
        
        # Epigenetic inputs
        epi_start = feat_dim + ont_dim + offt_dim
        input_4 = data[:, epi_start:epi_start+epi_dim].reshape((-1, 24, 4))
        input_5 = data[:, epi_start+epi_dim:epi_start+2*epi_dim].reshape((-1, 24, 4))
    
        return model.predict([input_1, input_2, input_3, input_4, input_5])

    data_indices = np.random.choice(len(concat_train_data), size=samples_size, replace=False)
    random_data = concat_train_data[data_indices]

    # Compute SHAP values with KernelExplainer
    shap_explainer = shap.KernelExplainer(shap_predictions, train_summary)
    shap_values = shap_explainer.shap_values(random_data, nsamples=perturbations_per_sample)

    np.save("shap_values.npy", shap_values)
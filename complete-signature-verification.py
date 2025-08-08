import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import moments_hu
from scipy.spatial.distance import cosine
from scipy.stats import skew, kurtosis
import mahotas as mt

class SignatureFeatureExtractor:
    """Class for extracting various types of features from signatures."""
    
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.cnn_model = self._load_cnn_model()

    def _load_cnn_model(self, model_name='vgg16'):
        """Load and prepare CNN model for feature extraction."""
        if model_name == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False)
            layer_name = 'block5_pool'
        else:
            base_model = ResNet50(weights='imagenet', include_top=False)
            layer_name = 'avg_pool'
            
        model = Model(inputs=base_model.input, 
                     outputs=base_model.get_layer(layer_name).output)
        return model

    def extract_cnn_features(self, img_path):
        """Extract CNN features from signature."""
        try:
            img = image.load_img(img_path, target_size=self.img_size, color_mode='rgb')
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = self.cnn_model.predict(img_array, verbose=0)
            return features.flatten()
        except Exception as e:
            print(f"Error extracting CNN features: {e}")
            return None

    def extract_texture_features(self, img_path):
        """Extract GLCM and LBP texture features."""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(32)

        # GLCM Features
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(img, distances, angles, symmetric=True, normed=True)
        
        glcm_features = []
        properties = ['contrast', 'dissimilarity', 'homogeneity', 
                     'energy', 'correlation', 'ASM']
        for prop in properties:
            glcm_features.extend(graycoprops(glcm, prop)[0])

        # LBP Features
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                             range=(0, n_points + 2))
        lbp_features = hist.astype(float) / (hist.sum() + 1e-7)

        return np.concatenate([glcm_features, lbp_features])

    def extract_geometric_features(self, img_path):
        """Extract geometric features from signature."""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(15)

        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros(15)

        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate basic shape features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w)/h if h != 0 else 0
        extent = float(area)/(w*h) if w*h != 0 else 0
        
        # Calculate Hu Moments
        moments = moments_hu(binary)
        
        # Calculate slant
        angle = cv2.minAreaRect(largest_contour)[-1]
        
        return np.concatenate([[area, perimeter, aspect_ratio, extent, angle], 
                             moments])

    def extract_structural_features(self, img_path):
        """Extract structural features using skeleton analysis."""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(20)

        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        skeleton = mt.thin(binary)
        
        # Find endpoints and junction points
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]])
        filtered = cv2.filter2D(skeleton.astype(float), -1, kernel)
        
        endpoints = np.sum(filtered == 11)
        junctions = np.sum(filtered >= 13)
        
        num_labels, labels = cv2.connectedComponents(255 - skeleton.astype(np.uint8))
        
        pixel_count = np.sum(skeleton)
        mean_intensity = np.mean(img[skeleton > 0]) if pixel_count > 0 else 0
        
        return np.array([endpoints, junctions, num_labels - 1,
                        pixel_count, mean_intensity])

class DynamicSignatureVerifier:
    """Class for dynamic signature verification using LSTM."""
    
    def __init__(self, sequence_length=64, feature_dim=32):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.model = self._build_lstm_model()

    def _build_lstm_model(self):
        """Build and compile LSTM model."""
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), 
                         input_shape=(self.sequence_length, self.feature_dim)),
            Dropout(0.2),
            Bidirectional(LSTM(64)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        return model

    def extract_dynamic_features(self, img_path):
        """Extract dynamic features from signature."""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((self.sequence_length, self.feature_dim))

        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        skeleton = cv2.ximgproc.thinning(binary)
        
        contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, 
                                     cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return np.zeros((self.sequence_length, self.feature_dim))

        all_points = np.vstack([contour.squeeze() for contour in contours 
                              if len(contour) > 0])
        all_points = all_points[all_points[:, 0].argsort()]

        features = []
        for i in range(len(all_points) - 1):
            point = all_points[i]
            next_point = all_points[i + 1]

            velocity = np.linalg.norm(next_point - point)
            angle = np.arctan2(next_point[1] - point[1], 
                             next_point[0] - point[0])
            
            window = img[max(point[1]-5, 0):min(point[1]+6, img.shape[0]),
                        max(point[0]-5, 0):min(point[0]+6, img.shape[1])]
            
            local_features = [
                np.mean(window),
                np.std(window),
                velocity,
                angle,
                point[0]/img.shape[1],
                point[1]/img.shape[0],
            ]
            
            features.append(local_features)

        features = np.array(features)
        if len(features) > self.sequence_length:
            indices = np.linspace(0, len(features)-1, self.sequence_length, dtype=int)
            features = features[indices]
        else:
            padding = np.zeros((self.sequence_length - len(features), 
                              len(features[0]) if len(features) > 0 else self.feature_dim))
            features = np.vstack([features, padding]) if len(features) > 0 else padding

        expanded_features = np.zeros((self.sequence_length, self.feature_dim))
        expanded_features[:, :features.shape[1]] = features
        
        return expanded_features

class SignatureVerifier:
    """Main class for signature verification combining all approaches."""
    
    def __init__(self):
        self.feature_extractor = SignatureFeatureExtractor()
        self.dynamic_verifier = DynamicSignatureVerifier()
        self.thresholds = {
            'cnn': 0.7,
            'texture': 0.6,
            'geometric': 0.65,
            'structural': 0.6,
            'dynamic': 0.5
        }
        self.weights = {
            'cnn': 0.3,
            'texture': 0.2,
            'geometric': 0.2,
            'structural': 0.15,
            'dynamic': 0.15
        }

    def calculate_similarity(self, vector1, vector2):
        """Calculate cosine similarity between feature vectors."""
        if len(vector1.shape) == 2:
            vector1 = vector1.flatten()
        if len(vector2.shape) == 2:
            vector2 = vector2.flatten()
            
        if np.all(vector1 == 0) or np.all(vector2 == 0):
            return 0
            
        return 1 - cosine(vector1, vector2)

    def verify_signature(self, real_path, test_path):
        """Verify if test signature matches real signature."""
        # Extract all features
        features_real = {
            'cnn': self.feature_extractor.extract_cnn_features(real_path),
            'texture': self.feature_extractor.extract_texture_features(real_path),
            'geometric': self.feature_extractor.extract_geometric_features(real_path),
            'structural': self.feature_extractor.extract_structural_features(real_path)
        }
        
        features_test = {
            'cnn': self.feature_extractor.extract_cnn_features(test_path),
            'texture': self.feature_extractor.extract_texture_features(test_path),
            'geometric': self.feature_extractor.extract_geometric_features(test_path),
            'structural': self.feature_extractor.extract_structural_features(test_path)
        }

        # Calculate similarities
        similarities = {}
        for feature_type in ['cnn', 'texture', 'geometric', 'structural']:
            similarities[feature_type] = self.calculate_similarity(
                features_real[feature_type],
                features_test[feature_type]
            )

        # Get dynamic similarity
        dynamic_features_real = self.dynamic_verifier.extract_dynamic_features(real_path)
        dynamic_features_test = self.dynamic_verifier.extract_dynamic_features(test_path)
        dynamic_diff = np.absolute(dynamic_features_real - dynamic_features_test)
        dynamic_diff = np.expand_dims(dynamic_diff, axis=0)
        similarities['dynamic'] = float(self.dynamic_verifier.model.predict(dynamic_diff)[0][0])

        # Calculate weighted score
        weighted_score = sum(similarities[k] * self.weights[k] 
                           for k in self.weights.keys())

        # Check all thresholds
        threshold_verdict = all(
            similarities[k] >= self.thresholds[k] 
            for k in self.thresholds.keys()
        )

        return {
            'similarities': similarities,
            'weighted_score': weighted_score,
            'threshold_verdict': threshold_verdict,
            'is_genuine': weighted_score >= 0.7 and threshold_verdict
        }

def main():
    """Main function to run signature verification on a dataset."""
    # Initialize verifier
    verifier = SignatureVerifier()
    
    # Set up paths
    base_dir = r'C:\Users\PMLS\train'
    csv_path = r'C:\Users\PMLS\OneDrive\Documents\train\signatures.csv'
    signatures_df = pd.read_csv(csv_path)
    
    results = []
    
    # Process each signature pair
    for index, row in signatures_df.iterrows():
        print(f"\nProcessing signature pair {index + 1}")
        
        real_path = os.path.join(base_dir, row['real_signature'])
        test_path = os.path.join(base_dir, row['submitted_signature'])
        
        if not os.path.exists(real_path) or not os.path.exists(test_path):
            print(f"Warning: Image path does not exist for index {index}")
            continue
            
        # Verify signature
        result = verifier.verify_signature(real_path, test_path)
        
        # Add paths to result
        result['real_signature'] = real_path
        result['test_signature'] = test_path
        
        # Add to results list
        results.append(result)
        
        # Print progress
        print(f"Signature pair {index + 1} processed:")
        print(f"Similarities: {result['similarities']}")
        print(f"Weighted Score: {result['weighted_score']:.3f}")
        print(f"Is Genuine: {result['is_genuine']}")
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    output_path = r'C:\Users\PMLS\OneDrive\Documents\train\signature_verification_results.csv'
    results_df.to_csv(output_path, index=False)
    
    # Print summary statistics
    print("\nVerification Complete!")
    print(f"Total signatures processed: {len(results_df)}")
    print(f"Signatures classified as genuine: {sum(results_df['is_genuine'])}")
    print(f"Verification rate: {(sum(results_df['is_genuine']) / len(results_df)) * 100:.2f}%")

if __name__ == "__main__":
    main()

# Import required libraries
import os
import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import moments_hu
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from scipy.spatial.distance import cosine
from scipy.stats import skew, kurtosis
import mahotas as mt
from PIL import Image

class SignatureVerifier:
    def __init__(self, model_name='vgg16'):
        """Initialize the signature verifier with selected model."""
        self.model = self.load_cnn_model(model_name)
        self.img_size = (224, 224)
        self.thresholds = {
            'cnn': 0.7,
            'texture': 0.6,
            'geometric': 0.65,
            'structural': 0.6,
            'statistical': 0.65
        }
        
    def load_cnn_model(self, model_name):
        """Load and prepare CNN model for feature extraction."""
        if model_name == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False)
            layer_name = 'block5_pool'
        elif model_name == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False)
            layer_name = 'avg_pool'
        else:
            raise ValueError("Unsupported model")
            
        model = Model(inputs=base_model.input, 
                     outputs=base_model.get_layer(layer_name).output)
        return model

    def preprocess_image(self, img_path):
        """Preprocess image for CNN."""
        try:
            img = image.load_img(img_path, target_size=self.img_size, color_mode='rgb')
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            return preprocess_input(img_array)
        except Exception as e:
            print(f"Error preprocessing image {img_path}: {e}")
            return None

    def extract_texture_features(self, img_path):
        """Extract texture features using GLCM and LBP."""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(32)

        # GLCM Features
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(img, distances, angles, symmetric=True, normed=True)
        
        glcm_features = []
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 
                     'correlation', 'ASM']
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

        # Binarize image
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate basic shape features
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(15)

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w)/h if h != 0 else 0
        extent = float(area)/(w*h) if w*h != 0 else 0
        
        # Calculate Hu Moments
        moments = moments_hu(binary)
        
        # Calculate slant angle
        angle = cv2.minAreaRect(largest_contour)[-1]
        
        # Combine all geometric features
        geometric_features = np.array([
            area, perimeter, aspect_ratio, extent, angle,
            *moments  # Add Hu moments
        ])
        
        return geometric_features

    def extract_structural_features(self, img_path):
        """Extract structural features using skeleton analysis."""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(20)

        # Binarize and create skeleton
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        skeleton = mt.thin(binary)
        
        # Find endpoints and junction points
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]])
        filtered = cv2.filter2D(skeleton.astype(float), -1, kernel)
        
        endpoints = np.sum(filtered == 11)  # One neighbor
        junctions = np.sum(filtered >= 13)  # Three or more neighbors
        
        # Calculate loop features
        num_labels, labels = cv2.connectedComponents(255 - skeleton.astype(np.uint8))
        
        # Calculate other structural properties
        pixel_count = np.sum(skeleton)
        mean_intensity = np.mean(img[skeleton > 0]) if pixel_count > 0 else 0
        
        structural_features = np.array([
            endpoints, junctions, num_labels - 1,  # -1 for background
            pixel_count, mean_intensity
        ])
        
        return structural_features

    def extract_statistical_features(self, img_path):
        """Extract statistical features from signature."""
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros(10)

        # Calculate basic statistical measures
        mean = np.mean(img)
        std = np.std(img)
        skewness = skew(img.ravel())
        kurt = kurtosis(img.ravel())
        
        # Calculate zoning features
        h, w = img.shape
        zones = []
        for i in range(2):
            for j in range(2):
                zone = img[i*h//2:(i+1)*h//2, j*w//2:(j+1)*w//2]
                zones.append(np.mean(zone))
        
        # Combine features
        statistical_features = np.array([
            mean, std, skewness, kurt, *zones
        ])
        
        return statistical_features

    def calculate_similarity(self, vector1, vector2):
        """Calculate cosine similarity between feature vectors."""
        if len(vector1.shape) == 2:
            vector1 = vector1.flatten()
        if len(vector2.shape) == 2:
            vector2 = vector2.flatten()
            
        if np.all(vector1 == 0) or np.all(vector2 == 0):
            return 0
            
        return 1 - cosine(vector1, vector2)

    def verify_signature(self, real_path, submitted_path):
        """Verify if submitted signature matches the real signature."""
        # Preprocess images
        preprocessed_real = self.preprocess_image(real_path)
        preprocessed_submitted = self.preprocess_image(submitted_path)
        
        if preprocessed_real is None or preprocessed_submitted is None:
            print(f"Skipping pair: {real_path}, {submitted_path} due to preprocessing error.")
            return {
                'similarities': {k: 0 for k in self.thresholds.keys()},
                'weighted_score': 0,
                'threshold_verdict': False,
                'is_genuine': False
            }
        
        # Extract all features
        features_real = {
            'cnn': self.model.predict(preprocessed_real, verbose=0).flatten(),
            'texture': self.extract_texture_features(real_path),
            'geometric': self.extract_geometric_features(real_path),
            'structural': self.extract_structural_features(real_path),
            'statistical': self.extract_statistical_features(real_path)
        }
        
        features_submitted = {
            'cnn': self.model.predict(preprocessed_submitted, verbose=0).flatten(),
            'texture': self.extract_texture_features(submitted_path),
            'geometric': self.extract_geometric_features(submitted_path),
            'structural': self.extract_structural_features(submitted_path),
            'statistical': self.extract_statistical_features(submitted_path)
        }
        
        # Calculate similarities
        similarities = {}
        for feature_type in self.thresholds.keys():
            similarities[feature_type] = self.calculate_similarity(
                features_real[feature_type],
                features_submitted[feature_type]
            )
        
        # Calculate weighted verdict
        weights = {
            'cnn': 0.3,
            'texture': 0.2,
            'geometric': 0.2,
            'structural': 0.2,
            'statistical': 0.1
        }
        weighted_score = sum(similarities[k] * weights[k] for k in weights.keys())
        threshold_verdict = weighted_score >= 0.75
        
        return {
            'similarities': similarities,
            'weighted_score': weighted_score,
            'threshold_verdict': threshold_verdict,
            'is_genuine': threshold_verdict
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
        submitted_path = os.path.join(base_dir, row['submitted_signature'])
        
        if not os.path.exists(real_path) or not os.path.exists(submitted_path):
            print(f"Warning: Image path does not exist for index {index}")
            continue
            
        # Verify signature
        verification_result = verifier.verify_signature(real_path, submitted_path)
        
        # Add paths to result
        verification_result['real_signature'] = real_path
        verification_result['submitted_signature'] = submitted_path
        
        # Add to results list
        results.append(verification_result)
        
        # Print progress
        print(f"Signature pair {index + 1} processed:")
        print(f"Similarities: {verification_result['similarities']}")
        print(f"Weighted Score: {verification_result['weighted_score']:.3f}")
        print(f"Is Genuine: {verification_result['is_genuine']}")
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    output_path = r'C:\Users\PMLS\OneDrive\Documents\train\signature_verification_results.csv'
    results_df.to_csv(output_path, index=False)
    
    # Print summary statistics
    print("\nVerification Complete!")
    print(f"Total signatures processed: {len(results_df)}")
    print(f"Signatures classified as genuine: {sum(results_df['is_genuine'])}")
    print(f"Verification rate: {(sum(results_df['is_genuine'])/len(results_df))*100:.2f}%")
    
    # Print average similarities
    print("\nAverage Similarity Scores:")
    for feature_type in verifier.thresholds.keys():
        avg_similarity = np.mean([r['similarities'][feature_type] for r in results])
        print(f"{feature_type}: {avg_similarity:.3f}")

if __name__ == "__main__":
    main()

import keras
from keras.models import load_model
from keras.models import Model
import scipy as sp
from keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2


path_model = '/content/gdrive/My Drive/tomato/tuned_model.h5'

class CAM_gen:

    def __init__(self):
        self.model = load_model(path_model)
        self.xcp = self.model.get_layer('xception')
        self.last_conv_layer = self.xcp.get_layer('block14_sepconv2_act')
        self.cut_model = Model(inputs = self.xcp.input, outputs = self.last_conv_layer.output)
        self.class_weight = self.model.layers[-3].get_weights()[0]
        self.class_weight_last = self.model.layers[-1].get_weights()[0]
        self.img_path=""
        self.img = None
 
    def proc_img(self, img_path):

        img = image.load_img(img_path, target_size=(299, 299))
        img_tensor = image.img_to_array(img) # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0) 
        img_tensor /= 255.

        return img_tensor

    def get_CAM(self, img_path):

        or_img = self.proc_img(img_path)
        self.img = or_img[0]
        self.img_path = img_path
        conv_output = self.cut_model.predict(or_img)
        conv_output = conv_output[0,:,:,:]
        output_1 = np.dot(conv_output, self.class_weight)
        final_output = np.dot(output_1, self.class_weight_last)
        cam_features = sp.ndimage.zoom(final_output, (29.9, 29.9, 1), order=1).reshape(299,299)

        return cam_features

    def plot_CAM(self, cam_features):

        heatmap = np.maximum(cam_features, 0)
        heatmap /= np.max(heatmap)


        img = np.uint8(255 * self.img)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        merged=  self.img + heatmap * 0.5  
        plt.imshow(merged)


        if __name__ == "__main__":
            test = CAM_gen()
            cam_test = test.get_CAM("/content/gdrive/My Drive/tomato/test.jpg")
            test.plot_CAM(cam_test)

from sklearn.neighbors import KNeighborsClassifier
import cv2

class FindNemo:
    def __init__(self, train_image):
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.X_train, self.Y_train = self.convert_image_to_dataset(train_image)
        self.knn.fit(self.X_train, self.Y_train)

    def convert_image_to_dataset(self, image):
        image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        pixels_list_hsv = image_hsv.reshape(-1, 3)
        light_orange = (1, 190, 200)
        dark_orange = (18, 255, 255)
        light_white = (0, 0, 200)
        dark_white = (145, 60, 255)
        light_black = (205, 255, 0)
        dark_black = (255, 255, 25)
        mask_orange = cv2.inRange(image_hsv, light_orange, dark_orange)
        mask_white = cv2.inRange(image_hsv, light_white, dark_white)
        mask_black = cv2.inRange(image_hsv, light_black, dark_black)
        final_mask = mask_orange + mask_white + mask_black
        x_train = pixels_list_hsv / 255
        y_train = final_mask.reshape(-1, ) // 255
        return x_train, y_train

    def remove_background(self, test_image):
        test_image = cv2.resize(test_image, (0, 0), fx=0.25, fy=0.25)
        image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        X_test = test_image_hsv.reshape(-1, 3) / 255
        Y_predict = self.knn.predict(X_test)
        output = Y_predict.reshape(test_image_hsv.shape[:2])
        final_result = cv2.bitwise_and(test_image, test_image, mask=output)
        return final_result



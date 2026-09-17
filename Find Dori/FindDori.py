from sklearn.neighbors import KNeighborsClassifier
import cv2

class FindDori:
    def __init__(self, train_image):
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.X_train, self.Y_train = self.convert_image_to_dataset(train_image)
        self.knn.fit(self.X_train, self.Y_train)

    def convert_image_to_dataset(self, image):
        image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        pixels_list_hsv = image.reshape(-1, 3)
        light_blue = (100, 180, 200)
        dark_blue = (125, 255, 255)
        light_black = (115, 190, 0)
        dark_black = (125, 255, 100)
        light_yellow = (20, 140, 180)
        dark_yellow = (31, 200, 255)
        mask_blue = cv2.inRange(image, light_blue, dark_blue)
        mask_black = cv2.inRange(image, light_black, dark_black)
        mask_yellow = cv2.inRange(image, light_yellow, dark_yellow)
        final_mask = mask_blue + mask_black + mask_yellow
        x_train = pixels_list_hsv / 255
        y_train = final_mask.reshape(-1, ) // 255
        return x_train, y_train

    def remove_background(self, test_image):
        test_image = cv2.resize(test_image, (0, 0), fx=0.25, fy=0.25)
        test_image_hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
        X_test = test_image_hsv.reshape(-1, 3) / 255
        Y_predict = self.knn.predict(X_test)
        output = Y_predict.reshape(test_image_hsv.shape[:2])
        final_result = cv2.bitwise_and(test_image, test_image, mask=output)
        return final_result



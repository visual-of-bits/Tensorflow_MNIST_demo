import tensorflow as tf
import cv2
import numpy as np

if __name__ == '__main__':

    #MNIST 입력으로 받기
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    #이미지를 0-1로 스케일하기
    x_train, x_test = x_train / 255.0, x_test / 255.0


    #Keras 모델 구성
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    #모델의 컴파일
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #모델의 학습
    model.fit(x_train, y_train, epochs=3)

    #학습된 모델에서 데이터 받기
    x_answer=model.predict(x_train)


    #계산 결과 출력과 정답 맞추기
    for i in range(0,100):
        #계산 결관 추출
        answer=np.where(x_answer[i]==x_answer[i].max())[0][0]

        #이미지 확대
        scaled_image = cv2.resize(x_train[i],(500, 500), interpolation=cv2.INTER_NEAREST)

        #MNIST 이미지를 파일에 쓰기
        cv2.imwrite("input/"+str(i)+".png",scaled_image*255)

        #출력 보여주기
        cv2.imshow("Answer:"+str(answer),scaled_image)
        print("출력:"+str(answer)+"  정답: "+str(y_train[i])+"  ",end="")
    
        #정답 맞추기
        if(answer==y_train[i]):
            print("맞음")
        else:
            print("틀렸음")

        #타이머
        cv2.waitKey(1000)
    
        #윈도우 지우기
        cv2.destroyAllWindows()
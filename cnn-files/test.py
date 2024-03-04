import tensorflow as ts
import skimage as ski

loaded_model = ts.keras.saving.load_model(
    r"conv_model___Date_Time_2024_03_03__13_53_46___Loss_1.0283106565475464___Accuracy_0.5825982093811035.keras")

# test the model for one prediction
image_data = ski.io.imread("ar3.jpg")
new_image = image_data.tolist()
result = loaded_model.predict([new_image])[0]

max_result = max(result)

if result[0] == max_result:
    print("Angry")
elif result[1] == max_result:
    print("Disgust")
elif result[2] == max_result:
    print("Fear")
elif result[3] == max_result:
    print("Happy")
elif result[4] == max_result:
    print("Natural")
elif result[5] == max_result:
    print("Sad")
elif result[6] == max_result:
    print("Surprised")

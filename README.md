# Coding3_element2
video link: https://studio.youtube.com/video/gJ6S8fIM1_w/edit

This project is based on Deepdream.
## Topic
Let me start by introducing my topic, In China there are some families who prefer sons who will take a medicine that can change the gender of the baby when they get pregnant, they think it will make the girl in the womb into a boy, but in fact it will only give birth to a girl who will have both male and female features.
The sea hare is a representative hermaphrodite, so I wanted to combine baby and sea hare, so that baby would have the features of a sea hare. In this way to satirise son-preference family.

Then I found Deepdream, which generates images that would be similar to the texture of the sea hare, So I want to replace the texture in deepdream with the sea hare texture.


## Code
For the first notebook (01_download_images.ipynb) I used the code from week 3 to download images of sea hares from Pinterest and make a dataset. But later I found that only the sea hare dataset was not enough, I had to add features of sea hare in order for the CNN model to recognise baby as sea hare as well, so I added some baby images to the dataset as well.

In the second notebook（02_creat_cnn_model.ipynb） I made a model using CNN which got the features of the sea hare by recognising and classifying it. I want to use this model to replace the pre-trained model in deepdream. But I found that if I replace the pre-trained model in deepdream, the generated image will be full of noise, so I replaced the weights of the pre-trained model and I saved the weights of this model here.

In the third notebook (03_deepdream.ipynb) I used deepdream's code where I changed the weights of the pre-trained model, but the actual changes didn't vary much. I used a image of a baby and then I changed the number of different layers here and each layer outputs a different result.

The fourth notebook is in this folder, I used the code from the third notebook in the week 7 file and wanted to animate the change from baby to sea hare using stable diffusion. The animation required interpolation between the two images, so I also combined the code from this site. The final animation generated looks like this.





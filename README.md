# face_recognition
This application uses the siamese architecture for face recognition.
It uses mtcnn for face detection and extraction and facenet model for converting the face to its encoding.
The app captures live feed and performs face recognition ont he current frame when you press **d**, quits by pressing **q**.

## How it works:
First you need pictures of people's faces in the name format: "name_randomNumber" in the face_rec_assets folder,
but if you don't have that you can create it live by editing the code in **main** function, then everytime you press **c**, that person's face will be added to the folder.

After we have the faces, In the live feed when **d** is pressed, the face captured is converted into its representation, and the representation of the saved faces are retrieved,
the L1 norm between each of them is computed, the results are aggregated by **mean** or **min**, if the difference is lower than some threshold, the face is of the person being compared to, otherwise not.

## Next steps:
create an interface that makes it easier to do things like adding new people to the database or removing.
Enable the already avalible ability of inputing files and detecting faces in it via the interface.
Enable the already availible ability of inputing videos and detecting faces in them via the interface.

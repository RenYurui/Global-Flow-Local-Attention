wget -O dataset/animation_skeletons.rar https://1drv.ws/u/s\!ArirMHnmz_frlBeL5s8lQ6eCfsNb
cd dataset
unrar x animation_skeletons.rar
cd danceFashion/train_256
unrar x train_alphapose.rar
unrar x train_video2d.rar
rm *.rar
cd ..
cd test_256
unrar x train_alphapose.rar
unrar x train_video2d.rar
rm *.rar
cd ..
cd ..
cd iPER/train_256
unrar x train_alphapose.rar
unrar x train_video2d.rar
rm *.rar
cd ..
cd test_256
unrar x train_alphapose.rar
unrar x train_video2d.rar
rm *.rar



wget -O dataset/animation_skeletons.rar https://1drv.ms/u/s!ArirMHnmz_frlBfWgrPlnCFBJYiw?e=LOiHEI
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



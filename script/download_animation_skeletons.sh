wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=17Fx56eJF_4-ky9GC8srh4sMuDwI3W0zq' -O dataset/animation_skeletons.rar
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



source("R\\materialmodifier.R")
im = im_load("D:\\cjm\\code\\my project\\DWT\\materialmodifier\\notes\\face.jpg")
# apply the shine effect
im2 = modif(im, effect = "shine", strength = 3.0) # may take some seconds
plot(im2) # see the result
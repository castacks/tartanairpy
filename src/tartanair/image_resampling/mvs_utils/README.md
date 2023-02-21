# mvs_utils
Utility codes for multi-view stereo.


### Conventions.
* An image's range starts from coordinate (0, 0) to (W, H), including (W, H).
* An image's pixels have coordinates starting from (0.5, 0.5) to (W-0.5, H-0.5). The total number of pixels is still W x H.

* When talking about coordinates, it is always in [x, y] order.
* When talking about shape, it is always in [H, W] order.
* When talking about size, it is always in [W, H] order. (To be compatible with the likes of OpenCV). 

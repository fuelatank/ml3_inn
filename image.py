
#import cards as _c
import os

import PIL.Image as _Img, PIL.ImageTk as _ImgTk#, PIL.ImageFont as _Imf, PIL.ImageDraw as _Imd

PATH = 'innovation\\innovation-dogma.mse-style\\'
backs = ['blue_back', 'red_back', 'green_back', 'yellow_back', 'purple_back']
icons = {'crown': 'crown_big', 'leaf': 'leaf_big', 'lightbulb': 'lamp_big',
'castle': 'fort_big', 'factory': 'plant_big', 'clock': 'clock_big'}
nums = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']

left = (26, 199, 375)
top = (27, 223)
pos = [(0, 0), (0, 1), (1, 1), (2, 1)]

#font = _Imf.truetype('timesbd.ttf', 36)

def getImage(name, age, icon, color):
	if os.path.exists('images\\'+name+'.jpg'):
		return _Img.open('images\\'+name+'.jpg')
	back = _Img.open(PATH+backs[color]+'.jpg')
	age = _Img.open(PATH+nums[age-1]+'.PNG')
	img = back.copy()
	img.paste(age, box=(462, 28, 494, 60))
	for e, i in enumerate(icon):
		if i:
			ic = _Img.open(PATH+icons[i]+'.JPG')
			mask = None
		else:
			ic = _Img.open(PATH+'custom_big.PNG')
			maskImg = _Img.open(PATH+'custom_mask.PNG')
			mask = maskImg.split()[0]
		l = left[pos[e][0]]
		t = top[pos[e][1]]
		img.paste(ic, (l, t, l+126, t+126), mask)
		if mask:
			maskImg.close()
		ic.close()
	draw = _Imd.Draw(img)
	draw.text((157, 23), name, font=font)
	age.close()
	back.close()
	img.save('images\\'+name+'.jpg')
	return img

def getTkImage(name, age, icon, color):
	if os.path.exists('images\\reduced\\'+name+'.jpg'):
		return _ImgTk.PhotoImage(_Img.open('images\\reduced\\'+name+'.jpg'))
	img = getImage(name, age, icon, color)
	rimg = img.reduce(3)
	rimg.save('images\\reduced\\'+name+'.jpg')
	return _ImgTk.PhotoImage(rimg)

if __name__ == '__main__':
	img = getImage('Masonry', 1, ['castle', None, 'castle', 'castle'], 3)
	img.show()
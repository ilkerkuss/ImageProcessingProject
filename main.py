from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import *
import cv2
import numpy as np
from skimage import data, filters
from matplotlib import pyplot as plt
from skimage.transform import swirl
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour

window = Tk()
window.title("Image Processing")
window.geometry('950x400')

tab_control = ttk.Notebook(window)
tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)
tab4 = ttk.Frame(tab_control)
tab5 = ttk.Frame(tab_control)
tab6 = ttk.Frame(tab_control)
tab7 = ttk.Frame(tab_control)
tab8 = ttk.Frame(tab_control)

tab_control.add(tab1, text='Görüntü Aç/Kaydet')
tab_control.add(tab2, text='Görüntü İyileştirme')
tab_control.add(tab3, text='Histogram')
tab_control.add(tab4, text='Uzaysal Dönüşüm')
tab_control.add(tab5, text='Yoğunluk Dönüşümü')
tab_control.add(tab6, text='Morfolojik İşlemler')
tab_control.add(tab7, text='Video İşleme')
tab_control.add(tab8, text='Active Contour/Fotoğraf Efekt')
#-----------------------------------------------------------Görüntü açma/gösterme/kaydetme
lbl1 = Label(tab1, text= 'Dosya Konumu :    ',bg='orange',fg='blue')
txt = Entry(tab1,width=50)
lbl2 = Label(tab1, text= 'Dosyayı Kaydet :  ')
txt2 = Entry(tab1,width=50)

lbl2.grid(column=0, row=1)
txt.grid(column=1, row=0)
lbl1.grid(column=0, row=0)
txt2.grid(column=1,row=1)
txt.focus()

def openFile():#görüntü belleğe alır
    messagebox.showinfo('Bilgilendirme', 'Görüntü Başarıyla Belleğe alındı')
    return cv2.imread(txt.get())

def showFile():#görüntüyü gösterir
    image=cv2.imread(txt.get())
    cv2.imshow('Görüntü',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def saveFile():#görüntüyü kaydeder
    image=cv2.imread(txt.get())
    cv2.imwrite(txt2.get()+str('.png'),image )


btn = Button(tab1, text="Görüntü Yükle", command=openFile)
btn.grid(column=2, row=0)
btn2 = Button(tab1, text="Görüntü Göster", command=showFile)
btn2.grid(column=3, row=0)
btn3 = Button(tab1, text="Görüntü Kaydet", command=saveFile)
btn3.grid(column=2, row=1)

#-----------------------------------------------------------morphology
def createKernel(selection):#5x5 kernel yaratır
    if (selection == 1):#rectangle kernel
        return cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    elif (selection == 2):#elliptical kernel
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    elif (selection == 3):#cross shaped kernel
        return cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    elif(selection == 4):
        return np.ones((5,5,np.uint8))

def selectedMorph(image,selection,kernel):#seçime göre morphlogy işlemi yapar ve kaydeder
    image_convert = cv2.cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if(selection==1):
        opening = cv2.morphologyEx(image_convert, cv2.MORPH_OPEN, kernel)
        cv2.imshow('opening örneği',cv2.cvtColor(opening,cv2.COLOR_BGR2RGB))
        cv2.imwrite('opening.png', cv2.cvtColor(opening,cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif (selection==2):
        erosion= cv2.erode(image_convert, kernel, iterations = 1)
        cv2.imshow('erosion örneği',cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB))
        cv2.imwrite('erosion.png',cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif(selection==3):
        closing = cv2.morphologyEx(image_convert, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('closing örneği',cv2.cvtColor(closing, cv2.COLOR_BGR2RGB))
        cv2.imwrite('closing.png',cv2.cvtColor(closing, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif(selection==4):
        gradient = cv2.morphologyEx(image_convert, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow('gradient örneği', cv2.cvtColor(gradient, cv2.COLOR_BGR2RGB))
        cv2.imwrite('gradient.png', cv2.cvtColor(gradient, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif(selection==5):
        dilation = cv2.dilate(image_convert,kernel,iterations = 1)
        cv2.imshow('dilation örneği', cv2.cvtColor(dilation, cv2.COLOR_BGR2RGB))
        cv2.imwrite('dilation.png', cv2.cvtColor(dilation, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif(selection==6):
        tophat = cv2.morphologyEx(image_convert, cv2.MORPH_TOPHAT, kernel)
        cv2.imshow('tophat örneği', cv2.cvtColor(tophat, cv2.COLOR_BGR2RGB))
        cv2.imwrite('tophat.png', cv2.cvtColor(tophat, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif(selection==7):
        blackhat = cv2.morphologyEx(image_convert, cv2.MORPH_BLACKHAT, kernel)
        cv2.imshow('blackhat örneği', cv2.cvtColor(blackhat, cv2.COLOR_BGR2RGB))
        cv2.imwrite('blackhat.png', cv2.cvtColor(blackhat, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print('seçim yapılmadı.')#se

def submitMorph():#morphology işlemini başlatır
    image=cv2.imread(txtMorph.get())
    seçim = selectionMorph.get()
    kernelSeçim=selectionKernel.get()
    selectedMorph(image,seçim,createKernel(kernelSeçim))


selectionMorph=IntVar()
selectionKernel=IntVar()

labelMorph=Label(tab6, text= 'Dosya Konumu :    ',bg='orange',fg='blue')
txtMorph = Entry(tab6,width=50)
radMo1 = Radiobutton(tab6,text='Opening', value=1,variable=selectionMorph)
radMo2 = Radiobutton(tab6,text='Erosion', value=2,variable=selectionMorph)
radMo3 = Radiobutton(tab6,text='Closing', value=3,variable=selectionMorph)
radMo4 = Radiobutton(tab6,text='Gradient', value=4,variable=selectionMorph)
radMo5 = Radiobutton(tab6,text='Dilation', value=5,variable=selectionMorph)
radMo6 = Radiobutton(tab6,text='Top Hat', value=6,variable=selectionMorph)
radMo7 = Radiobutton(tab6,text='Black Hat', value=7,variable=selectionMorph)
radMo8 = Radiobutton(tab6,text='Rectangle Kernel', value=1,variable=selectionKernel)
radMo9 = Radiobutton(tab6,text='Elliptical Kernel', value=2,variable=selectionKernel)
radMo10 = Radiobutton(tab6,text='Cross-Shaped Kernel', value=3,variable=selectionKernel)
btn4=Button(tab6, text="Görüntüle", command=submitMorph)

labelMorph.grid(column=0,row=0)
txtMorph.grid(column=1, row=0)
radMo1.grid(column=0, row=1)
radMo2.grid(column=0, row=2)
radMo3.grid(column=0, row=3)
radMo4.grid(column=0, row=4)
radMo5.grid(column=0, row=5)
radMo6.grid(column=0, row=6)
radMo7.grid(column=0, row=7)
radMo8.grid(column=1, row=1)
radMo9.grid(column=1, row=2)
radMo10.grid(column=1, row=3)
btn4.grid(column=2, row=0)
#-----------------------------------------------------------görüntü iyileştirme
def selectedEnc(image,selectionEnc):#yapılan seçime göre filtreleme işlemleri yapar

    if (selectionEnc == 1):
        sobel=filters.sobel(image)
        cv2.imshow('Orijinal Görüntü', image)
        cv2.imshow('Sobel Filter',sobel)
        cv2.imwrite('Sobel.png', sobel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif (selectionEnc == 2):
        blur = cv2.blur(image,(5,5))
        cv2.imshow('Orijinal Görüntü', image)
        cv2.imshow('Blur Filter', blur)
        cv2.imwrite('Blur.png', blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif (selectionEnc == 3):
        prewitt = filters.prewitt(image)
        cv2.imshow('Orijinal Görüntü', image)
        cv2.imshow('Prewitt Filter', prewitt)
        cv2.imwrite('Prewitt.png', prewitt)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif (selectionEnc == 4):
        laplacian = cv2.Laplacian(image,cv2.CV_64F)
        cv2.imshow('Orijinal Görüntü', image)
        cv2.imshow('Laplacian Filter', laplacian)
        cv2.imwrite('Laplacian.png', laplacian)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif (selectionEnc == 5):
        keskinlestirici = np.array([[-1, -1, -1, -1, -1],
                                    [-1, 2, 2, 2, -1],
                                    [-1, 2, 8, 2, -1],
                                    [-1, 2, 2, 2, -1],
                                    [-1, -1, -1, -1, -1]]) / 8.0

        sharpening = cv2.filter2D(image,-1,keskinlestirici)
        cv2.imshow('Orijinal Görüntü', image)
        cv2.imshow('Sharpening Filter', sharpening)
        cv2.imwrite('Sharpening.png', sharpening)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif (selectionEnc == 6):
        kabartma = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])

        embossing = cv2.filter2D(image, -1, kabartma)
        cv2.imshow('Orijinal Görüntü', image)
        cv2.imshow('Embossing Filter', embossing)
        cv2.imwrite('Embossing.png', embossing)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif (selectionEnc == 7):
        canny = cv2.Canny(image, 50, 200)
        cv2.imshow('Orijinal Görüntü', image)
        cv2.imshow('Canny Filter', canny)
        cv2.imwrite('Canny.png', canny)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif (selectionEnc == 8):
        gaussianBlur = cv2.GaussianBlur(image, (13, 13), 0)
        cv2.imshow('Orijinal Görüntü', image)
        cv2.imshow('Gaussian Blur Filter', gaussianBlur)
        cv2.imwrite('Gaussian Blur.png', gaussianBlur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif (selectionEnc == 9):
        bilateral = cv2.bilateralFilter(image, 13, 70, 50)
        cv2.imshow('Orijinal Görüntü', image)
        cv2.imshow('Bilateral Filter', bilateral)
        cv2.imwrite('Bilateral.png', bilateral)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif (selectionEnc == 10):
        medianBlur = cv2.medianBlur(image, ksize=7)
        cv2.imshow('Orijinal Görüntü', image)
        cv2.imshow('Median Blur Filter', medianBlur)
        cv2.imwrite('Median Blur.png', medianBlur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def submitEnc():#filtreleme işlemini başlatır
    image = cv2.imread(txtEnc.get())
    seçimEnc = selectionEnc.get()
    print(seçimEnc)
    selectedEnc(image,seçimEnc)

selectionEnc=IntVar()

labelEnc=Label(tab2, text= 'Dosya Konumu :    ',bg='orange',fg='blue')
txtEnc = Entry(tab2,width=50)
radEnc1 = Radiobutton(tab2,text='Sobel Filter', value=1,variable=selectionEnc)
radEnc2 = Radiobutton(tab2,text='Blur Filter', value=2,variable=selectionEnc)
radEnc3 = Radiobutton(tab2,text='Prewitt Filter', value=3,variable=selectionEnc)
radEnc4 = Radiobutton(tab2,text='Laplacian Filter', value=4,variable=selectionEnc)
radEnc5 = Radiobutton(tab2,text='Sharpening Filter', value=5,variable=selectionEnc)
radEnc6 = Radiobutton(tab2,text='Embossing Filter', value=6,variable=selectionEnc)
radEnc7 = Radiobutton(tab2,text='Canny Filter', value=7,variable=selectionEnc)
radEnc8 = Radiobutton(tab2,text='Gaussian Blur Filter', value=8,variable=selectionEnc)
radEnc9 = Radiobutton(tab2,text='Bilateral Filter', value=9,variable=selectionEnc)
radEnc10 = Radiobutton(tab2,text='Median Blur Filter', value=10,variable=selectionEnc)
btn5=Button(tab2, text="Görüntüle", command=submitEnc)

labelEnc.grid(column=0,row=0)
txtEnc.grid(column=1, row=0)
radEnc1.grid(column=0, row=1)
radEnc2.grid(column=0, row=2)
radEnc3.grid(column=0, row=3)
radEnc4.grid(column=0, row=4)
radEnc5.grid(column=0, row=5)
radEnc6.grid(column=1, row=1)
radEnc7.grid(column=1, row=2)
radEnc8.grid(column=1, row=3)
radEnc9.grid(column=1, row=4)
radEnc10.grid(column=1, row=5)
btn5.grid(column=2, row=0)
#-----------------------------------------------------------Histogram işlemleri
def getHistogram(image):#verilen görüntünün histogramını görüntüler
    plt.hist(image.ravel(),256,[0,256])
    plt.show()

def equalizedHist(image):#verilen imageyi color gray yapar ve o görüntünün histogramını eşitler geriye eşitlenmiş görüntüyü döndürür ve kaydeder
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equ=cv2.equalizeHist(gray)
    cv2.imshow('Equalized Image',equ)
    cv2.imwrite('Histogram Equalized Image.jpg',equ)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def submitHist():#histogram oluşturmayı başlatır
    image = cv2.imread(txtHist.get())
    getHistogram(image)

def submitEquHist():#histogram eşitlemeyi başlatır
    image = cv2.imread(txtHist.get())
    equalizedHist(image)

labelHist=Label(tab3, text= 'Dosya Konumu :    ',bg='orange',fg='blue')
txtHist=Entry(tab3,width=50)
btn6=Button(tab3, text="Histogram Oluştur", command=submitHist)
btn7=Button(tab3, text="Görüntü Histogramı Eşitle", command=submitEquHist)

labelHist.grid(column=0,row=0)
txtHist.grid(column=1, row=0)
btn6.grid(column=2, row=0)
btn7.grid(column=3, row=0)


#-----------------------------------------------------------Uzaysal dönüşüm işlemleri
def rotateImage(image):
    img_rotated=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('Rotated Image',img_rotated)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27: #esc to quit
            cv2.destroyAllWindows()
            break
        elif k == ord('d') or k==ord('D'): #right arrow
            img_rotated = cv2.rotate(img_rotated, cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow('Rotated Image',img_rotated)
            cv2.waitKey(1)
            continue
        elif k == ord('a') or k==('A'): #left arrow
            img_rotated = cv2.rotate(img_rotated, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow('Rotated Image',img_rotated)
            cv2.waitKey(1)
            continue

def flipImage(image):
    img_flipped=cv2.flip(image,0)
    cv2.imshow('Flipped Image',img_flipped)
    cv2.waitKey(1)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # esc to quit
            cv2.destroyAllWindows()
            break
        elif k == ord('w') or k == ('W'):  # flip horizontal
            img_flipped = cv2.flip(img_flipped, 0)
            # cv2.imwrite('right rotated image', img_rotate_right)
            cv2.imshow('Flipped Image', img_flipped)
            cv2.waitKey(1)
            continue
        elif k == ord('s') or k == ('S'):  # flip vertically
            img_flipped = cv2.flip(img_flipped, 1)
            # cv2.imwrite('right rotated image', img_rotate_right)
            cv2.imshow('Flipped Image', img_flipped)
            cv2.waitKey(1)
            continue

def cropImage(image,startX,endX,startY,endY):
    cropped_image=image[startY:endY,startX:endX]
    cv2.imshow('Cropped Image',cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def reSize(image,nWidth,nHeight):
    oWidth=image.shape[1]
    oHeight=image.shape[0]

    if(oHeight*oWidth>nWidth*nHeight):
        resize_image=cv2.resize(image,(nWidth,nHeight),interpolation=cv2.INTER_AREA)
        cv2.imshow('Interpolation Shrink',resize_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif(oHeight*oWidth<nWidth*nHeight):
        resize_image = cv2.resize(image, (nWidth, nHeight), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('Cubic Zooming', resize_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def reSized(image,percentage):
    resize_image = cv2.resize(image,None, fx=percentage/100,fy=percentage/100, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('Interpolation Percantage', resize_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def projectedImage(image):
    rows, cols = image.shape[:2]
    src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    dst_points = np.float32([[0, 0], [cols - 1, 0], [int(0.33 * cols), rows - 1],[int(0.66 * cols), rows - 1]])

    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    img_output = cv2.warpPerspective(image, projective_matrix, (cols, rows))

    cv2.imshow('Projected Image', img_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def swirlImage(image):
    swirled = swirl(image, rotation=0, strength=10, radius=150)
    cv2.imshow('Swirled Image',swirled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def submitRotate():
    messagebox.showinfo('Bilgilendirme', 'Görüntüyü Sola (A)- Sağa (D) tuşlarını kullanarak döndürebilirsiniz.\n Çıkış Yapmak için ESC tuşunu kullanınız.')
    image = cv2.imread(txtSpace.get())
    cpy=image.copy()
    rotateImage(cpy)

def submitFlip():
    messagebox.showinfo('Bilgilendirme', 'Görüntüyü Yukarı (W)- Aşağı (S) tuşlarını kullanarak çevirebilirsiniz.\n Çıkış Yapmak için ESC tuşunu kullanınız.')
    image = cv2.imread(txtSpace.get())
    cpy=image.copy()
    flipImage(cpy)

def submitCrop():
    image = cv2.imread(txtSpace.get())
    cpy=image.copy()
    startX=txtstartX.get()
    endX=txtendX.get()
    startY=txtstartY.get()
    endY=txtendY.get()
    cropImage(image,int(startX),int(endX),int(startY),int(endY))

def submitProjected():
    image = cv2.imread(txtSpace.get())
    cpy = image.copy()
    projectedImage(cpy)

def submitSwirl():
    image = cv2.imread(txtSpace.get())
    cpy = image.copy()
    swirlImage(cpy)

def submitResized():
    image = cv2.imread(txtSpace.get())
    cpy = image.copy()
    percentage=txtPercentage.get()
    reSized(cpy,int(percentage))

def submitResizeCoordinate():
    image = cv2.imread(txtSpace.get())
    cpy = image.copy()
    nWidth=txtnWidth.get()
    nHeight=txtnHeight.get()
    reSize(cpy, int(nWidth),int(nHeight))


labelSpace=Label(tab4, text= 'Dosya Konumu :    ',bg='orange',fg='blue')
txtSpace=Entry(tab4,width=50)
btn8=Button(tab4, text="Görüntü Döndür", command=submitRotate)
btn9=Button(tab4, text="Görüntü Çevir", command=submitFlip)
btn10=Button(tab4, text="Projeksiyon Görüntüsü", command=submitProjected)
btn11=Button(tab4, text="Swirl Görüntüsü", command=submitSwirl)
labelstartX=Label(tab4, text= 'X başlangıç :  ',bg='orange',fg='blue')
txtstartX=Entry(tab4,width=10)
labelendX=Label(tab4, text= 'X bitiş :  ',bg='orange',fg='blue')
txtendX=Entry(tab4,width=10)


labelstartY=Label(tab4, text= 'Y başlangıç :   ',bg='orange',fg='blue')
txtstartY=Entry(tab4,width=10)
labelendY=Label(tab4, text= 'Y bitiş :  ',bg='orange',fg='blue')
txtendY=Entry(tab4,width=10)
btn12=Button(tab4, text="Kırp", command=submitCrop)

labelResizePercentage=Label(tab4, text= 'Ölçeklendirilecek Yüzdeyi Giriniz :  ',bg='orange',fg='blue')
txtPercentage=Entry(tab4,width=10)
btn13=Button(tab4, text="Yüzde ile Ölçeklendir", command=submitResized)

labelnWidth=Label(tab4, text= 'Ölçeklendirilecek Genişlik Giriniz :  ',bg='orange',fg='blue')
txtnWidth=Entry(tab4,width=10)
labelnHeight=Label(tab4, text= 'Ölçeklendirilecek Yükseklik Giriniz :  ',bg='orange',fg='blue')
txtnHeight=Entry(tab4,width=10)
btn14=Button(tab4, text="Girdi ile Ölçeklendir", command=submitResizeCoordinate)

labelSpace.grid(column=0,row=0)
txtSpace.grid(column=1, row=0)
btn8.grid(column=2, row=0)
btn9.grid(column=3, row=0)
btn10.grid(column=4, row=0)
btn11.grid(column=5,row=0)

labelstartX.grid(column=0,row=1)
txtstartX.grid(column=1,row=1)
labelendX.grid(column=2,row=1)
txtendX.grid(column=3,row=1)

labelstartY.grid(column=0,row=2)
txtstartY.grid(column=1,row=2)
labelendY.grid(column=2,row=2)
txtendY.grid(column=3,row=2)
btn12.grid(column=4,row=2)

labelResizePercentage.grid(column=0,row=3)
txtPercentage.grid(column=1,row=3)
btn13.grid(column=2,row=3)

labelnWidth.grid(column=0,row=4)
txtnWidth.grid(column=1,row=4)
labelnHeight.grid(column=0,row=5)
txtnHeight.grid(column=1,row=5)
btn14.grid(column=2,row=5)

#-----------------------------------------------------------Yoğunluk dönüşüm işlemleri
def submitLog():
    image = cv2.imread(txtInt.get())
    logTransformation(image)

def submitGamma():
    image = cv2.imread(txtInt.get())
    gammaTranformation(image,float(txtGamma.get()))

def submitLinearTrans():
    image = cv2.imread(txtInt.get())
    piecewiseLinearTransformation(image,int(txtR1.get()),int(txtS1.get()),int(txtR2.get()),int(txtS2.get()))

def submitSelectCanal():
    image = cv2.imread(txtInt.get())
    selectedCanal(image,selectionInt.get())

def logTransformation(image):

    c = 255 / (np.log(1 + np.max(image)))
    log_transformed = c * np.log(1 + image)

    log_transformed = np.array(log_transformed, dtype=np.uint8)

    cv2.imshow('log_transformed.jpg', log_transformed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gammaTranformation(image,gammaValue):

    gamma_corrected = np.array(255 * (image / 255) ** (gammaValue), dtype='uint8')

    cv2.imshow('image',gamma_corrected)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def piecewiseLinearTransformation(image,r1,s1,r2,s2):

    def pixelVal(pix, r1, s1, r2, s2):
        if (0 <= pix and pix <= r1):
            return (s1 / r1) * pix
        elif (r1 < pix and pix <= r2):
            return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
        else:
            return ((255 - s2) / (255 - r2)) * (pix - r2) + s2


    pixelVal_vec = np.vectorize(pixelVal)

    contrast_stretched = pixelVal_vec(image, r1, s1, r2, s2)


    cv2.imshow('Contrast Stretch', contrast_stretched)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def selectedCanal(image,selectionInt):
    if(selectionInt==1):
        blueImage=image.copy()
        blueImage[:, :, 0] = 0
        blueImage[:, :, 1] = 0
        cv2.imshow('Mavi Kanal Örneği',cv2.cvtColor(blueImage,cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif (selectionInt==2):
        greenImage = image.copy()
        greenImage[:, :, 0] = 0
        greenImage[:, :, 2] = 0
        cv2.imshow('Yeşil Kanal Örneği', cv2.cvtColor(greenImage, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif(selectionInt==3):
        redImage = image.copy()
        redImage[:, :, 1] = 0
        redImage[:, :, 2] = 0
        cv2.imshow('Kırmızı Kanal Örneği', cv2.cvtColor(redImage, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        messagebox.showinfo('Bilgilendirme','Renk Kanalı Seçimi Yapılmadı.')


labelInt=Label(tab5, text= 'Dosya Konumu :    ',bg='orange',fg='blue')
txtInt=Entry(tab5,width=50)
btn15=Button(tab5, text="Log Dönüşümü ", command=submitLog)
labelGamma=Label(tab5, text= 'Gamma Değeri  :    ',bg='orange',fg='blue')
txtGamma=Entry(tab5,width=10)
btn16=Button(tab5, text="Gamma Dönüşümü", command=submitGamma)
labelR1=Label(tab5, text= 'r1(Girdi Yoğunluğu 1) :    ',bg='orange',fg='blue')
txtR1=Entry(tab5,width=10)
labelR2=Label(tab5, text= 'r2(Girdi Yoğunluğu 2) :    ',bg='orange',fg='blue')
txtR2=Entry(tab5,width=10)
labelS1=Label(tab5, text= 's1(Çıktı Yoğunluğu 1) :    ',bg='orange',fg='blue')
txtS1=Entry(tab5,width=10)
labelS2=Label(tab5, text= 's2(Çıktı Yoğunluğu 2) :    ',bg='orange',fg='blue')
txtS2=Entry(tab5,width=10)
btn17=Button(tab5, text="Linear Dönüşüm", command=submitLinearTrans)

selectionInt=IntVar()
radInt1 = Radiobutton(tab5,text='Mavi Kanal', value=1,variable=selectionInt)
radInt2 = Radiobutton(tab5,text='Yeşil Kanal', value=2,variable=selectionInt)
radInt3 = Radiobutton(tab5,text='Kırmızı Kanal', value=3,variable=selectionInt)
btn18=Button(tab5, text="Kanal Görüntüle", command=submitSelectCanal)

labelInt.grid(column=0,row=0)
txtInt.grid(column=1, row=0)
btn15.grid(column=2, row=0)

labelGamma.grid(column=0,row=1)
txtGamma.grid(column=1,row=1)
btn16.grid(column=2, row=1)

labelR1.grid(column=0,row=2)
txtR1.grid(column=1,row=2)
labelR2.grid(column=2,row=2)
txtR2.grid(column=3,row=2)

labelS1.grid(column=0,row=3)
txtS1.grid(column=1,row=3)
labelS2.grid(column=2,row=3)
txtS2.grid(column=3,row=3)
btn17.grid(column=4, row=3)

radInt1.grid(column=0,row=4)
radInt2.grid(column=1,row=4)
radInt3.grid(column=2,row=4)
btn18.grid(column=3,row=4)

#-----------------------------------------------------------Video işleme
def submitVideo():
    cap = cv2.VideoCapture(txtVideo.get())
    videoCanny(cap)

def videoCanny(video):
    cap = video
    while True:
        _, frame = cap.read()
        canny = cv2.Canny(frame, 50, 200)
        if frame is None:
            break
        cv2.imshow('Canny Efekt Video', canny)

        if cv2.waitKey(1) and 0x00 == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

labelVideo=Label(tab7, text= 'Dosya Konumu :    ',bg='orange',fg='blue')
txtVideo=Entry(tab7,width=50)
btn19=Button(tab7, text="Efekt Uygula", command=submitVideo)

labelVideo.grid(column=0,row=0)
txtVideo.grid(column=1, row=0)
btn19.grid(column=2,row=0)

#-----------------------------------------------------------Active contour/fotoğraf efekt
def submitEfect():
    image = cv2.imread(txtEfect.get())
    appendEfect(image)

def submitContour():
    activeContour(selectionActive.get())

def activeContour(selectionContour):
    if (selectionContour == 1):
        img = data.astronaut()
        img = rgb2gray(img)
        s = np.linspace(0, 2 * np.pi, 400)
        r = 100 + 100 * np.sin(s)
        c = 220 + 100 * np.cos(s)
        init = np.array([r, c]).T

        snake = active_contour(gaussian(img, 3),
                               init, alpha=0.015, beta=10, gamma=0.001,coordinates='rc')

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])

        plt.show()
    elif (selectionContour == 2):
        img = data.text()

        r = np.linspace(136, 50, 100)
        c = np.linspace(5, 424, 100)
        init = np.array([r, c]).T

        snake = active_contour(gaussian(img, 1), init, boundary_condition='fixed',
                               alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1,coordinates='rc')

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])

        plt.show()
    elif (selectionContour == 3):
        img = data.camera()
        img = rgb2gray(img)

        s = np.linspace(0, 2 * np.pi, 400)
        r = 100 + 100 * np.sin(s)
        c = 220 + 100 * np.cos(s)
        init = np.array([r, c]).T

        snake = active_contour(gaussian(img, 3),
                               init, alpha=0.015, beta=10, gamma=0.001,coordinates='rc')

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])

        plt.show()

def appendEfect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_1 = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    color = cv2.bilateralFilter(image, d=9, sigmaColor=200, sigmaSpace=200)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    cv2.imshow('Orjinal Fotoğraf',image)
    cv2.imshow('Cartoon Efekt',cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


labelEfect=Label(tab8, text= 'Dosya Konumu :    ',bg='orange',fg='blue')
txtEfect=Entry(tab8,width=50)
btn20=Button(tab8, text="Efekt Uygula", command=submitEfect)

selectionActive=IntVar()
radActive1 = Radiobutton(tab8,text='Astronot Örneği', value=1,variable=selectionActive)
radActive2 = Radiobutton(tab8,text='Yazı Örneği', value=2,variable=selectionActive)
radActive3 = Radiobutton(tab8,text='Kameraman Örneği', value=3,variable=selectionActive)
btn21=Button(tab8, text="Active Contour Örnek", command=submitContour)

labelEfect.grid(column=0,row=0)
txtEfect.grid(column=1, row=0)
btn20.grid(column=2,row=0)
radActive1.grid(column=0,row=1)
radActive2.grid(column=1,row=1)
radActive3.grid(column=2,row=1)
btn21.grid(column=3,row=1)




tab_control.pack(expand=1, fill='both')
window.mainloop()


clc, clear all, close all
%Generare imagine
interval = 15;
pas_esantionare = 5;
latime_interval = 0.3;
theta = 30*2*pi/180;
img = generareImagine(theta);
%Blur(generare PSF)
sigma = 2;
[psf, A] = blurareImagine(pas_esantionare, sigma, img);
figure, imshow(psf, []), title('PSF');
figure, imshow(A), title('Imagine blurata');
%Gradient
% [Gx,Gy]=imgradientxy(A,'sobel');
% [Gt,Gdir]=imgradient(A,'sobel');
% figure, imshow(Gt,[min(Gt(:)),max(Gt(:))]), title('Gradientul imaginii');

%Distanta de la fiecare pixel la linie
d = distantaPunctLinie(A);
D1 = d(abs(d) <50);
I = A(abs(d) <50);
figure, plot(D1(:),I(:),'.'), title('ESF');
xlabel('Pozitie')
ylabel('Intensitate')
%figure, imshow(d)

%Medierea intensitatilor pe intervale de lungime egala(Estimare ESF)

ESF = estimareESF(interval, d, latime_interval, A);

figure, plot(-interval:latime_interval:interval, ESF), title('ESF normat');
xlabel('Pozitie')
ylabel('Intensitate')
%Filtrarea pentru reducerea zgomotului

[Eh, R, Ef, L] = estimareLSF(ESF);
figure, plot(Eh);
xlabel('Pozitie')
ylabel('Intensitate')
%Reducerea ratei de esantionare

figure, plot(R);
xlabel('Pozitie')
ylabel('Intensitate')
%Filtrul median

figure, plot(Ef);
xlabel('Pozitie')
ylabel('Intensitate')
%Diferentierea discreta

figure, plot(L), title('LSF');
xlabel('Pozitie')
ylabel('Intensitate normata')
%Ferestruirea
K = 0:(length(L)-1);
N = length(L);
W = 1 - ((K-1/2*(N-1))/(1/2*(N+1))).^2 ;
Lw=L.*W ;
%figure, plot(W)
figure,plot(Lw)
%FFT
F = fft(Lw);
f_normat = abs(F)/max(abs(F));
f = (0:length(Lw)-1)/length(Lw);
plot(f,f_normat), title('MTF')
xlabel('Frecventa normata')
ylabel('Modulatie')


%SNR
L = L/sum(L);
meanrow_psf = sum(psf);
[Y,I] = max(meanrow_psf);
Xt = (1:length(meanrow_psf))-I;
[C,D] = max(L);
Xe = ((1:length(L))-D)*pas_esantionare*latime_interval*2;
meanrow_psf = meanrow_psf/Y;
L = L/C;
figure, plot(Xt,meanrow_psf);
xlabel('Pozitie')
ylabel('Intensitate')
hold on
plot(Xe,L);
Li = interp1(Xe,L,Xt);
SNR = -10*log10(sum((meanrow_psf - Li).^2)/sum(meanrow_psf.^2));

%IRADON(estimare PSF)
n_unghiuri = 60;
theta_array = (pi/n_unghiuri):(pi/n_unghiuri):pi;

matrice_lsf = zeros(n_unghiuri,length(L));
matrice_esf = zeros(n_unghiuri,length(ESF));


for i = 1:n_unghiuri
    %Generare imagine
    imgNoua = generareImagine(theta_array(i));
    [psf, blurredImgNoua] = blurareImagine(pas_esantionare,sigma,imgNoua);
    %DistantaPunctLinie
    [distanta, unghi] = distantaPunctLinie(blurredImgNoua);
    %Estimare ESF
    ESF_nou = estimareESF(interval, distanta, latime_interval, blurredImgNoua);
    matrice_esf(i,:) = ESF_nou;
    %Estimare LSF
    [Eh, R, Ef, LSF_nou] = estimareLSF(ESF_nou);
    
    matrice_lsf(i,:) = LSF_nou;
    
end

for i = n_unghiuri:-1:1
    if any(isnan(matrice_lsf(i,:)))
       theta_array(i) = [];
       matrice_lsf(i,:) = [];
    end
end

theta_array_degrees = (theta_array.*180)/pi;

psf_estim = iradon(matrice_lsf.', theta_array_degrees);
imshow(psf_estim,[],'InitialMagnification', 30);

function [img] = generareImagine(theta)
    rows = 256*5;
    columns = 256*5;
    img = zeros(rows, columns, 'uint8');
    x1 = 0;
    x2 = 256;
    y1 = 256/2+32;
    y2 = -sin(theta)*(x2-x1) + y1;
    xCoords = [x1     0  256 x2]*5;
    yCoords = [y1 0 0 y2]*5;
    mask = poly2mask(xCoords, yCoords, rows, columns);
    img(mask) = 255;
end

function [psf, blurredImg] = blurareImagine(pas_esantionare,sigma,img)
    psf=fspecial('disk',sigma*pas_esantionare);
    %psf = hex_rer(sigma*pas_esantionare);
    blurred=imfilter(img,psf,'conv','symmetric');
    blurredImg=blurred(1:pas_esantionare:end,1:pas_esantionare:end);
    %figure, imshow(psf, []);
end

function [distanta, unghi] = distantaPunctLinie(blurredImg)
    threshold = edge(blurredImg, 'canny');
    threshold(1:2,:)=0;
    threshold(:,1:2)=0;
    %Regresie
    [y, x]=find(threshold==1);
    %plot(x,y,'.')
    n = length(x);
    b = (mean(y)*sum(x.^2)-mean(x)*sum(x.*y))/(sum(x.^2)-n*mean(x)^2);
    a = (sum(x.*y)-mean(x)*sum(y))/(sum(x.^2)-n*mean(x)^2);
    y = x.*a+b;
    %figure, imshow(blurredImg),
    %hold on, plot(x,y,'LineWidth',2), title('Dreapta de regresie'), xlabel('x'),ylabel('y');
    %Distanta de la punct la linie
    x = repmat(1:length(blurredImg(1,:)),length(blurredImg(:,1)),1);
    y = x';
    distanta=(x.*a-y+b)/sqrt(a^2+1);
    unghi = atan(a);
end

function [ESF_Estimat] = estimareESF(interval, distantaPunctLinie, latime_interval, blurredImg)
    [randuri, coloane] = size(distantaPunctLinie);
    i=1;
    medieValori=[];
    for val = -interval : latime_interval : interval %impartire ESF
        valoriContorizate = 0;
        sumaValori = 0;

        for rand = 1 : randuri
            for coloana = 1 : coloane
                if (distantaPunctLinie(rand, coloana) >= val && distantaPunctLinie(rand, coloana) < (val + latime_interval))
                    sumaValori = sumaValori + im2double(blurredImg(rand, coloana));
                    valoriContorizate = valoriContorizate + 1;
                end
            end
        end
        medieValori(i) = sumaValori / valoriContorizate;
        i=i+1;
    end
    ESF_Estimat = medieValori;
end

function [Eh, R, Ef, LSF] = estimareLSF(ESF_estimat)
   %Filtrarea pentru reducerea zgomotului
    H=[0.001099 0 -0.007080 0 0.026184 0 -0.077942 0 0.307739 0.5 0.307739 0 -0.077942 0 0.026184 0 -0.007080 0 0.001099];
    Eh=conv(ESF_estimat, H, "valid"); 
    %Reducerea ratei de esantionare
    R = Eh(1:2:end);
    %Filtrul median
    Ef = medfilt1(R);
    %Diferentierea discreta
    LSF=conv(Ef,[1 0 -1], "valid");
end

function [ker] = hex_rer(size)
    x = linspace(-1,1,size);
    test1 = abs(x < 0.866);
    test2 = abs(x' - 0.577*x) < 1;
    test3 = abs(x' + 0.577*x) < 1;
    test = test1 .* test2 .* test3;
    ker = double(test);
    ker = ker / sum(ker(:));
end

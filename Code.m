clc
clear all
close all
load EEG1_1c31;% loading data
Ts=2;% sampling period
Fs=250;%sampling frequency
[N,nu]=size(data)%obtain size of data
t=(1:N)*Ts;%generates time vector
s1 = fft(data(:,1),N);
s2 = fft(data(:,1),1024);s3 = fft(data(:,1),512);s4 = fft(data(:,1),256);
f=1:N;
freq=(1:N)*(Fs/N);
%freq2=(1:1024)*Fs/1024;freq3=(1:512)*Fs/512;freq4=(1:256)*Fs/256;
%DELTA 0 - 4
Fs = 250; % Sampling Frequency
Fpass = 0; % Passband Frequency
Fstop = 4; % Stopband Frequency
Dpass = 0.057501127785; %Passband Ripple
Dstop = 0.0001; % Stopband Attenuation
dens = 20; % Density Factor
% Calculate the order from the parameters using FIRPMORD.
[No, Fo, Ao, W] = firpmord([Fpass, Fstop]/(Fs/2), [1 0], [Dpass, Dstop]);
% Calculate the coefficients using the FIRPM function.
bd = firpm(No, Fo, Ao, W, {dens});
Hd = dfilt.dffir(bd);
%grpdelay(Hd3,2048,Fs); % plot group delay
D=mean(grpdelay(Hd))
ds1=filter(Hd,s1);
ds2=filter(Hd,s2);
ds3=filter(Hd,s3);
ds4=filter(Hd,s4);
x1=filter(Hd,data(:,1));
d1=fft(x1,N);
figure
plot(freq,ds1,freq,d1,'r')
title('DELTA')
legend('fft_filter','filter_fft')
%THETA 4 - 8
Fs = 250; % Sampling Frequency
Fstop1 = 3.5; % First Stopband Frequency
Fpass1 = 4; % First Passband Frequency
Fpass2 = 8; % Second Passband Frequency
Fstop2 = 8.5; % Second Stopband Frequency
Dstop1 = 0.0001; % First Stopband Attenuation
Dpass = 0.057501127785; % Passband Ripple
Dstop2 = 0.0001; % Second Stopband Attenuation
46
dens = 20; % Density Factor
% Calculate the order from the parameters using FIRPMORD.
[No, Fo, Ao, W] = firpmord([Fpass, Fstop]/(Fs/2), [1 0], [Dpass, Dstop]);
% Calculate the coefficients using the FIRPM function.
bt = firpm(No, Fo, Ao, W, {dens});
Ht = dfilt.dffir(bt);
%grpdelay(Hd3,2048,Fs); % plot group delay
D=mean(grpdelay(Ht))
ts1=filter(Ht,s1);
ts2=filter(Ht,s2);
ts3=filter(Ht,s3);
ts4=filter(Ht,s4);
x1=filter(Ht,data(:,1));
t1=fft(x1,N);
figure
plot(freq,ts1,freq,t1,'r',freq,s1,'m')
title('THETA')
legend('fft_filtre','filter_fft','fft_original')
2. Plotting fast fourier transform of the signal for different N values
clc
clear all
close all
load EEG1_1c31;% loading data
Ts=2;% sampling period
Fs=500;%sampling frequency
[N,nu]=size(data)%obtain size of data
t=(1:N)*Ts;%generates time vector
s1 = fft(data(:,1),N);
s2 = fft(data(:,1),1024);s3 = fft(data(:,1),512);s4 = fft(data(:,1),256);
f=1:N;
freq=(1:N)*Fs/N;
freq2=(1:1024)*Fs/1024;freq3=(1:512)*Fs/512;freq4=(1:256)*Fs/256;
%DELTA 0 - 4
Fs = 500; % Sampling Frequency
Fpass = 0; % Passband Frequency
Fstop = 4; % Stopband Frequency
Dpass = 0.057501127785; %Passband Ripple
Dstop = 0.0001; % Stopband Attenuation
dens = 20; % Density Factor
% Calculate the order from the parameters using FIRPMORD.
[No, Fo, Ao, W] = firpmord([Fpass, Fstop]/(Fs/2), [1 0], [Dpass, Dstop]);
% Calculate the coefficients using the FIRPM function.
b3 = firpm(No, Fo, Ao, W, {dens});
Hd3 = dfilt.dffir(b3);
%grpdelay(Hd3,2048,Fs); % plot group delay
D=mean(grpdelay(Hd3))
x=filter(Hd3,data(:,1));
%x=filter(Hd3,[data(:,1);zeros(D,1)]);% Append D zeros to the input data
%x=x(D+1:end);% Shift data to compensate for delay
d1 = fft(x,N);
47
d2 = fft(x,1024); d3 = fft(x,512); d4 = fft(x,256);
%THETA 4 - 8
Fs = 500; % Sampling Frequency
Fstop1 = 3.5; % First Stopband Frequency
Fpass1 = 4; % First Passband Frequency
Fpass2 = 8; % Second Passband Frequency
Fstop2 = 8.5; % Second Stopband Frequency
Dstop1 = 0.0001; % First Stopband Attenuation
Dpass = 0.057501127785; % Passband Ripple
Dstop2 = 0.0001; % Second Stopband Attenuation
dens = 20; % Density Factor
% Calculate the order from the parameters using FIRPMORD.
[No, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
0], [Dstop1 Dpass Dstop2])
% Calculate the coefficients using the FIRPM function.
b3 =firpm(No, Fo, Ao, W, {dens});
Hd3 = dfilt.dffir(b3);
%grpdelay(Hd3,2048,Fs); % plot group delay
D=mean(grpdelay(Hd3))
x=filter(Hd3,[data(:,1);zeros(D,1)]);% Append D zeros to the input data
x=x(D+1:end);% Shift data to compensate for delay
t1 = fft(x,N);
t2 = fft(x,1024); t3 = fft(x,512); t4 = fft(x,256);
%ALPHA 8 - 12
Fs = 500; % Sampling Frequency
Fstop1 = 7.5; % First Stopband Frequency
Fpass1 = 8; % First Passband Frequency
Fpass2 = 12; % Second Passband Frequency
Fstop2 = 12.5; % Second Stopband Frequency
Dstop1 = 0.0001; % First Stopband Attenuation
Dpass = 0.057501127785; % Passband Ripple
Dstop2 = 0.0001; % Second Stopband Attenuation
dens = 20; % Density Factor
% Calculate the order from the parameters using FIRPMORD.
[No, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
0], [Dstop1 Dpass Dstop2])
% Calculate the coefficients using the FIRPM function.
b3 =firpm(No, Fo, Ao, W, {dens});
Hd3 = dfilt.dffir(b3);
%grpdelay(Hd3,2048,Fs); % plot group delay
D=mean(grpdelay(Hd3))
x=filter(Hd3,[data(:,1);zeros(D,1)]);% Append D zeros to the input data
x=x(D+1:end);% Shift data to compensate for delay
a1 = fft(x,N);
a2 = fft(x,1024); a3 = fft(x,512); a4 = fft(x,256);
%BETA 12 - 30
48
Fs = 500; % Sampling Frequency
Fstop1 = 11.5; % First Stopband Frequency
Fpass1 = 12; % First Passband Frequency
Fpass2 = 30; % Second Passband Frequency
Fstop2 = 30.5; % Second Stopband Frequency
Dstop1 = 0.0001; % First Stopband Attenuation
Dpass = 0.057501127785; % Passband Ripple
Dstop2 = 0.0001; % Second Stopband Attenuation
dens = 20; % Density Factor
% Calculate the order from the parameters using FIRPMORD.
[No, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...0], [Dstop1 Dpass Dstop2])
% Calculate the coefficients using the FIRPM function.
b3 =firpm(No, Fo, Ao, W, {dens});
Hd3 = dfilt.dffir(b3);
%grpdelay(Hd3,2048,Fs); % plot group delay
D=mean(grpdelay(Hd3))
x=filter(Hd3,[data(:,1);zeros(D,1)]);% Append D zeros to the input data
x=x(D+1:end);% Shift data to compensate for delay
b1 = fft(x,N);
b2 = fft(x,1024); b3 = fft(x,512); b4 = fft(x,256);
figure
plot(freq,abs(s1).^2,freq,abs(d1).^2,'k',freq,abs(t1).^2,'c',freq,abs(a1).^2,'r',freq,abs(b1).^2,'m','linewidth',1.5)
title('N = size of data, 1068');
xlabel('Frequency (f)')
legend('Original Signal','Delta','Theta','Alpha','Beta');
grid on
axis tight
figure
subplot(3,1,1)
plot(freq2,abs(s2).^2,freq2,abs(d2).^2,'k',freq2,abs(t2).^2,'c',freq2,abs(a2).^2,'r',freq2,abs(b2).^2,'m','linewidth',1.5)
title('N = 1024');
xlabel('Frequency (f)')
legend('Original Signal','Delta','Theta','Alpha','Beta');
grid on
axis tight
subplot(3,1,2)
plot(freq3,abs(s3).^2,freq3,abs(d3).^2,'k',freq3,abs(t3).^2,'c',freq3,abs(a3).^2,'r',freq3,abs(b3).^2,'m','linewidth',1.5)
title('N = 512');
xlabel('Frequency (f)')
legend('Original Signal','Delta','Theta','Alpha','Beta');
grid on
axis tight
subplot(3,1,3)
plot(freq4,abs(s4).^2,freq4,abs(d4).^2,'k',freq4,abs(t4).^2,'c',freq4,abs(a4).^2,'r',freq4,abs(b4).^2,'m','linewidth',1.5)
title('N = 256');
xlabel('Frequency (f)')
legend('Original Signal','Delta','Theta','Alpha','Beta');
grid on
49
axis tight
figure
subplot(2,2,1)
plot(freq,abs(s1).^2,freq,abs(a1).^2,'r','linewidth',1.5);
title('N = size of data, 1068');
xlabel('Frequency (f)')
legend('Original Signal','Filtered Signal');
grid on
axis tight
subplot(2,2,2)
plot(freq2,abs(a2).^2,freq2,abs(y2).^2,'r','linewidth',1.5);
title('N = 1024');
xlabel('Frequency (f)')
legend('Original Signal','Filtered Signal');
grid on
axis tight
subplot(2,2,3)
plot(freq3,abs(a3).^2,freq3,abs(y3).^2,'r','linewidth',1.5);
title('N = 512');
xlabel('Frequency (f)')
legend('Original Signal','Filtered Signal');
grid on
axis tight
subplot(2,2,4)
plot(freq4,abs(a4).^2,freq4,abs(y4).^2,'r','linewidth',1.5);
title('N = 256');
xlabel('Frequency (f)')
legend('Original Signal','Filtered Signal');
grid on
axis tight
suptitle('Power Spectrum of ALPHA Band')
ed=abs(s1).^2 - abs(d1).^2;
et=abs(s1).^2 - abs(t1).^2;
ea=abs(s1).^2 - abs(a1).^2;
eb=abs(s1).^2 - abs(b1).^2;
freqed=freq>0 &freq< 4.5;
freqet=freq>3.5 &freq< 8.5;
freqea=freq>7.5 &freq< 12.5;
freqeb=freq>11.5 &freq< 30.5;
figure
plot(freq(freqed),ed(freqed),'k',freq(freqet),et(freqet),'c',freq(freqea),ea(freqea),'r',freq(freqeb),eb(freqeb),'m')
title('error')
xlabel('Frequency (f)')
legend('Delta','Theta','Alpha','Beta');
grid on
axis tight
3. To plot wavelet coefficients of normal and epileptic patients
load Z096.txt;
load S096.txt;
waveletFunction = 'db20';
[C,L] = wavedec(Z096,5,waveletFunction);
cD1 = detcoef(C,L,1); %NOISE
50
cD2 = detcoef(C,L,2); %GAMMA
cD3 = detcoef(C,L,3); %BETA
cD4 = detcoef(C,L,4); %ALPHA
cD5 = detcoef(C,L,5); %THETA
cA5 = appcoef(C,L,waveletFunction,5); %DELTA
D1 = wrcoef('d',C,L,waveletFunction,1) ;
D2 = wrcoef('d',C,L,waveletFunction,2); %GAMMA
D3 = wrcoef('d',C,L,waveletFunction,3); %BETA
D4 = wrcoef('d',C,L,waveletFunction,4); %ALPHA
D5 = wrcoef('d',C,L,waveletFunction,5);%THETA
A5 = wrcoef('a',C,L,waveletFunction,5);%DELTA
figure;
subplot(6,1,1); plot(D1);
ylabel('D1-noise');
title('normal');
subplot(6,1,2); plot(D2);
ylabel('D2-gamma');
subplot(6,1,3); plot(D3);
ylabel('D3-beta');
subplot(6,1,4); plot(D4);
ylabel('D4-alpha');
subplot(6,1,5); plot(D5);
ylabel('D5-theta');
subplot(6,1,6); plot(A5);
ylabel('A5-delta');
waveletFunction = 'db20';
[C1,L1] = wavedec(S096,5,waveletFunction);
cD1_ = detcoef(C1,L1,1); %NOISE
cD2_= detcoef(C1,L1,2); %GAMMA
cD3_ = detcoef(C1,L1,3); %BETA
cD4_ = detcoef(C1,L1,4); %ALPHA
cD5_ = detcoef(C1,L1,5); %THETA
cA5_ = appcoef(C1,L1,waveletFunction,5); %DELTA
D1_ = wrcoef('d',C1,L1,waveletFunction,1);
D2_ = wrcoef('d',C1,L1,waveletFunction,2); %GAMMA
D3_ = wrcoef('d',C1,L1,waveletFunction,3); %BETA
D4_ = wrcoef('d',C1,L1,waveletFunction,4); %ALPHA
D5_ = wrcoef('d',C1,L1,waveletFunction,5); %THETA
A5_ = wrcoef('a',C1,L1,waveletFunction,5);%DELTA
figure;
subplot(6,1,1); plot(D1_);
ylabel('D1-noise');
title('epileptic');
subplot(6,1,2); plot(D2_);
ylabel('D2-gamma');
subplot(6,1,3); plot(D3_);
ylabel('D3-beta');
subplot(6,1,4); plot(D4_);
ylabel('D4-alpha')
subplot(6,1,5); plot(D5_);
ylabel('D5-theta');
51
subplot(6,1,6); plot(A5_);
ylabel('A5-delta’);
4. Feature Extraction codes
a=max(D2);
b=min(D2);
n=a-b
sum=0;
for i=1:length(D2)-1
c=D2(i+1)-D2(i);
d=abs(c);
sum=sum+d;
end
m=log(sum);
h=log(n);
k=m/h;
min(D2)
max(D2)
mean(D2)
std(D2)
L2=length(D2)
for i=1:L2
x2=D2(i)*D2(i);
y2=x2*log(x2);
end
display(y2)
energy1=0;
energy2=0;
energy3=0;
energy4=0;
energy5=0;
energy6=0;
for i=1:length(D1)
energyN=D1(i)*D1(i);
energy1=energy1+energyN;
end
% display(energy1);
energyN=0;
for i=1:length(D2)
energyN=D2(i)*D2(i);
energy2=energy2+energyN;
end
display(energy2);
energyN=0;
for i=1:length(D3)
energy1=D3(i)*D3(i);
energy3=energy3+energyN;
end
% display(energy3);
energyN=0;
for i=1:length(D4)
energyN=D4(i)*D4(i);
energy4=energy4+energyN;
end
52
% display(energy4);
energyN=0;
for i=1:length(D5)
energyN=D5(i)*D5(i);
energy5=energy5+energyN;
end
% display(energy5);
energyN=0;
for i=1:length(A5)
energyN=A5(i)*A5(i);
energy6=energy6+energyN;
end
% display(energy6);
etotal=energy1+energy2+energy3+energy4+energy5+energy6;
E4=energy2/etotal;
5. Neural network code for classification
inputs = load (‘traindata.txt’);
inputs=inputs';
targets = load (‘target.txt)
targets=targets';
hiddenLayerSize = 15;
for i=1:30
net = feedforwardnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.trainFcn = 'trainrp'
[net,tr] = train(net,inputs,targets);
check=[-90.4923 85.0807 0.0145 20.7752 1.77E+06 0.1862 1.66E+01 ];
check=check';
outputs = net(check);
% outputs=net(inputs);
% errors = gsubtract(targets,outputs);
% performance = perform(net,targets,outputs)
% tInd = tr.testInd;
% tstOutputs = net(inputs(:,tInd));
% tstPerform = perform(net,targets(:,tInd),tstOutputs)
% figure, plotperform(tr)
% figure, plotconfusion(targets,outputs)
% % load('matlab.mat')
v = getx(net);
setwb(net,v);
end
display(outputs);

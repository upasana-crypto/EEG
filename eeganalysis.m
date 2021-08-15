 clc
clear all
close all
load EEG1_1c31;% loading data
Fs=250;%sampling frequency
Ts=1/Fs;% sampling period
[N,nu]=size(data)%obtain size of data
t=(1:N)*Ts;%generates time vector
 s1 = fft(data(:,1),N);
 s2 = fft(data(:,1),1024);s3 = fft(data(:,1),512);s4 = fft(data(:,1),256);

freq=(1:N)*Fs/N;
freq2=(1:1024)*Fs/1024;freq3=(1:512)*Fs/512;freq4=(1:256)*Fs/256;



%DELTA 0 - 4

Fs = 250;  % Sampling Frequency
Fpass = 0;               % Passband Frequency
Fstop = 4;               % Stopband Frequency
Dpass = 0.057501127785;  % Passband Ripple
Dstop = 0.0001;          % Stopband Attenuation
dens  = 20;              % Density Factor
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
  d2 = fft(x,1024); d3 = fft(x,512); d4 = fft(x,256);

  
 %THETA 4 - 8
 
Fs = 500;  % Sampling Frequency
Fstop1 = 3.5;             % First Stopband Frequency
Fpass1 = 4;               % First Passband Frequency
Fpass2 = 8;              % Second Passband Frequency
Fstop2 = 8.5;            % Second Stopband Frequency
Dstop1 = 0.0001;          % First Stopband Attenuation
Dpass  = 0.057501127785;  % Passband Ripple
Dstop2 = 0.0001;          % Second Stopband Attenuation
dens   = 20;              % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[No, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                          0], [Dstop1 Dpass Dstop2])
% Calculate the coefficients using the FIRPM function.
b3  = firpm(No, Fo, Ao, W, {dens});
Hd3 = dfilt.dffir(b3);
%grpdelay(Hd3,2048,Fs); % plot group delay
D=mean(grpdelay(Hd3))          
 x=filter(Hd3,[data(:,1);zeros(D,1)]);% Append D zeros to the input data
 x=x(D+1:end);% Shift data to compensate for delay
 t1 = fft(x,N);
  t2 = fft(x,1024); t3 = fft(x,512); t4 = fft(x,256);
 
  
%ALPHA 8 - 12

Fs = 500;  % Sampling Frequency
Fstop1 = 7.5;             % First Stopband Frequency
Fpass1 = 8;               % First Passband Frequency
Fpass2 = 12;              % Second Passband Frequency
Fstop2 = 12.5;            % Second Stopband Frequency
Dstop1 = 0.0001;          % First Stopband Attenuation
Dpass  = 0.057501127785;  % Passband Ripple
Dstop2 = 0.0001;          % Second Stopband Attenuation
dens   = 20;              % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[No, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                          0], [Dstop1 Dpass Dstop2])
% Calculate the coefficients using the FIRPM function.
b3  = firpm(No, Fo, Ao, W, {dens});
Hd3 = dfilt.dffir(b3);
%grpdelay(Hd3,2048,Fs); % plot group delay
D=mean(grpdelay(Hd3))          
 x=filter(Hd3,[data(:,1);zeros(D,1)]);% Append D zeros to the input data
 x=x(D+1:end);% Shift data to compensate for delay
 a1 = fft(x,N);
  a2 = fft(x,1024); a3 = fft(x,512); a4 = fft(x,256);
  
  
  %BETA 12 - 30
  
  Fs = 500;  % Sampling Frequency
Fstop1 = 11.5;             % First Stopband Frequency
Fpass1 = 12;               % First Passband Frequency
Fpass2 = 30;              % Second Passband Frequency
Fstop2 = 30.5;            % Second Stopband Frequency
Dstop1 = 0.0001;          % First Stopband Attenuation
Dpass  = 0.057501127785;  % Passband Ripple
Dstop2 = 0.0001;          % Second Stopband Attenuation
dens   = 20;              % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[No, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                          0], [Dstop1 Dpass Dstop2])
% Calculate the coefficients using the FIRPM function.
b3  = firpm(No, Fo, Ao, W, {dens});
Hd3 = dfilt.dffir(b3);
%grpdelay(Hd3,2048,Fs); % plot group delay
D=mean(grpdelay(Hd3))          
 x=filter(Hd3,[data(:,1);zeros(D,1)]);% Append D zeros to the input data
 x=x(D+1:end);% Shift data to compensate for delay
 b1 = fft(x,N);
  b2 = fft(x,1024); b3 = fft(x,512); b4 = fft(x,256);
  
  
  
  
  
  figure
  plot(freq,abs(s1).^2,freq,abs(d1).^2,'k',freq,abs(t1).^2,'c',freq,abs(a1).^2,'r',freq,abs(b1).^2,'m','linewidth',1.5)
  title('PSD graph for N(size of data)= 64');
 xlabel('Frequency (f)')
 ylabel('dB')
legend('Original Signal','Delta','Theta','Alpha','Beta');
 grid on
 axis tight
  
  
 figure
 subplot(3,1,1)
  plot(freq2,abs(s2).^2,freq2,abs(d2).^2,'k',freq2,abs(t2).^2,'c',freq2,abs(a2).^2,'r',freq2,abs(b2).^2,'m','linewidth',1.5)
 title('PSD graph for N(size of data)= 1024');
 xlabel('Frequency (f)')
legend('Original Signal','Delta','Theta','Alpha','Beta');
 grid on
 axis tight
 subplot(3,1,2)
  plot(freq3,abs(s3).^2,freq3,abs(d3).^2,'k',freq3,abs(t3).^2,'c',freq3,abs(a3).^2,'r',freq3,abs(b3).^2,'m','linewidth',1.5)
 title('PSD graph for N(size of data)= 512');
 xlabel('Frequency (f)')
 ylabel('dB')
legend('Original Signal','Delta','Theta','Alpha','Beta');
 grid on
 axis tight
 subplot(3,1,3)
  plot(freq4,abs(s4).^2,freq4,abs(d4).^2,'k',freq4,abs(t4).^2,'c',freq4,abs(a4).^2,'r',freq4,abs(b4).^2,'m','linewidth',1.5)
 title('N = 256');
 xlabel('Frequency (f)')
 ylabel('dB')
legend('Original Signal','Delta','Theta','Alpha','Beta');
 grid on
 axis tight
 
%  %figure
%  subplot(2,2,1)
% plot(freq,abs(s1).^2,freq,abs(a1).^2,'r','linewidth',1.5);
% title('N = size of data, 1068');
% xlabel('Frequency (f)')
% legend('Original Signal','Filtered Signal');
% grid on
% axis tight
% subplot(2,2,2)
% plot(freq2,abs(a2).^2,freq2,abs(y2).^2,'r','linewidth',1.5);
% title('N = 1024');
% xlabel('Frequency (f)')
% legend('Original Signal','Filtered Signal');
% grid on
% axis tight
% subplot(2,2,3)
% plot(freq3,abs(a3).^2,freq3,abs(y3).^2,'r','linewidth',1.5);
% title('N = 512');
% xlabel('Frequency (f)')
% legend('Original Signal','Filtered Signal');
% grid on
% axis tight
% subplot(2,2,4)
% plot(freq4,abs(a4).^2,freq4,abs(y4).^2,'r','linewidth',1.5);
% title('N = 256');
% xlabel('Frequency (f)')
% legend('Original Signal','Filtered Signal');
% grid on
% axis tight
% suptitle('Power Spectrum of ALPHA Band')
ed=abs(s1).^2 - abs(d1).^2;
et=abs(s1).^2 - abs(t1).^2;
ea=abs(s1).^2 - abs(a1).^2;
eb=abs(s1).^2 - abs(b1).^2;
freqed=freq>0 & freq < 4.5;
freqet=freq>3.5 & freq < 8.5;
freqea=freq>7.5 & freq < 12.5;
freqeb=freq>11.5 & freq < 30.5;
figure
plot(freq(freqed),ed(freqed),'k',freq(freqet),et(freqet),'c',freq(freqea),ea(freqea),'r',freq(freqeb),eb(freqeb),'m')
title('error in fft_ filter and filter_ fft')
xlabel('Frequency (f)')
ylabel('dB')
legend('Delta','Theta','Alpha','Beta');
 grid on
 axis tight

%set(gca,'position',[0 0 1 1],'units','normalized')
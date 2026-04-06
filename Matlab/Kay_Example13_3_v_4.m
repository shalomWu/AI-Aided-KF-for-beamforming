function [h_pred,h_hat_Classic] = Kay_Example13_3_v_4(N,SNRdB,c,a,q_1,m,q_2,dt)

%clear;
%N=4001;
%t=0:1:N-1;
p=1;
%BPSK_vec=[-1  1];
QPSK_vec=[1+1i 1-1i -1+1i -1-1i]/sqrt(2);

%% This is the channel taken from the QuADriga simulation
h_Quad_0=c*sqrt(N/sum(abs(c).^2));
% h_Quad_0=h_Quad_0';

%h_Quad_1=c1*sqrt(N/sum(abs(c1).^2));
% h_Quad_1=h_Quad_1';

% figure;plot(real(h_Quad_0));%hold on;plot(real(h_Quad_1),'-r')
% grid on;
%
%% Channel Coefficents. Will vay in time eventually
% h_0=(randn(1,N)+1i*randn(1,N))*sqrt(0.5);%*ones(1,N);%sqrt(0.25)*ones(1,N);%sqrt(0.4)*ones(1,N)+sqrt(0.25)*sin(0.01*t);%zeros(1,N);
% h_1=(randn(1,N)+1i*randn(1,N))*sqrt(0.5);%;%sqrt(0.5)*ones(1,N);%ones(1,N);%
h_0=sqrt(1)*h_Quad_0(1:N);%sqrt(0.4)*ones(1,N)+sqrt(0.25)*sin(0.01*t);%zeros(1,N);
h_1=zeros(1,N);%sqrt(1)*h_Quad_1(1:N);%ones(1,N);%

h_2=zeros(1,N);%sqrt(0.25)*ones(1,N)+sqrt(0.25)*sin(0.01*t);%zeros(1,N);
h_3=zeros(1,N);%sqrt(0.25)*ones(1,N);
h_4=zeros(1,N);%sqrt(0.25)*ones(1,N);%;

H=[h_0];%;h_1;h_2;h_3;h_4];
%H=H; % Normalizing for SNR purposes

%% This should be our BPSK\QPSK signal later on
%v=sqrt(2)*cos(0.1*t);



%% Channel output
%y=h_0(3:end).*v(3:end)+h_1(2:end-1).*v(2:end-1)+h_2(1:end-2).*v(1:end-2);
% figure;
% plot(y,'-r')
% hold on
% plot(y_m,'.b')
% grid on



%% Adding noise
%SNRdB=0:2:40;
SER=zeros(1,length(SNRdB));
SNRdB_Theory=zeros(1,length(SNRdB));
P_err_linear=zeros(1,length(SNRdB));
SNR=10.^(SNRdB./10);
MSE=zeros(length(SNRdB),p);
MSE_pred=zeros(length(SNRdB),p);
theta=pi/4;

for k=1:length(SNRdB) %/vector for SNR
    %disp(['SNR: ', num2str(SNRdB(k)),'dB']);
    %rng(12345);
    %v=randsrc(1,N,QPSK_vec);
    v=randsrc(1,N,QPSK_vec);

    %v1=zeros(1,N);
    %V=[v(p:end)',v(p-1:end-1)',v(p-2:end-2)',v(p-3:end-3)',v(1:end-(p-1))']; % This is for the estimation of a FIR filter in time domain
    %V=[v(p:end)',v(p-1:end-1)',v(p-2:end-2)',v(p-3:end-3)',v(1:end-(p-1))'];
    %V=[randsrc(1,N-p,QPSK_vec)',randsrc(1,N-p,QPSK_vec)',randsrc(1,N-p,QPSK_vec)',randsrc(1,N-p,QPSK_vec)',randsrc(1,N-p,QPSK_vec)'];
    %V=[randn(1,N-p)',randn(1,N-p)',randn(1,N-p)',randn(1,N-p)',randn(1,N-p)'];
    V=v(1:end)';%,v1',v1',v1',v1'];
    %V=[randsrc(1,N,QPSK_vec)',randsrc(1,N,QPSK_vec)',randsrc(1,N,QPSK_vec)',randsrc(1,N,QPSK_vec)',randsrc(1,N,QPSK_vec)'];


   % y=diag(conj(V)*G)';
    y=diag(conj(V)*H)';
    %y=v.*H;

    %%Option from QuadRiga
    % y_Quad=h_Quad.*v;
    % y_Quad=y_Quad(1:end-(p-1));


    rho=1./sqrt(SNR(k));
    %x=y_Quad+rho*randn(1,length(y));
    w=(randn(1,length(y))+1i*randn(1,length(y)))/sqrt(2); %AWGN

    x=y+rho*w;

    % figure
    % scatter(real(V1),imag(V1),'b*')
    % hold on
    % scatter(real(V),imag(V),'rX','LineWidth', 4)
    % grid on

    %% Implenting the VKF for the Channel
    h_hat  = exp(-1i * theta)*ones(p,N);
    h_pred = exp(-1i * theta)*ones(p,N);
    h_hat_Classic=ones(p,N);
    % A=[0.999,0,0;
    %     0,0.999,0;
    %     0,0,0.999];

    %dt   = 5;%0.09;                  % sampling interval
    q_h  = q_1;                  % process‐noise on h
    q_v  = q_2;                  % process‐noise on v (tune separately if you like)
    rho2 = rho;              % measurement noise variance

    %--- Build augmented matrices ---
    A = [ a*eye(p),     dt*eye(p);
        zeros(p),   a*eye(p)   ];

    Q = blkdiag( q_h*eye(p),  q_v*eye(p) );

    %--- Initializations ---
    N = length(x);
    M = m * eye(2*p);          % P(1|1) on [h; v]
    x_hat = zeros(2*p, N);     % full state estimates
    % init h to first H-column, v to zero
    x_hat(:,1) = [ H(:,1); zeros(p,1) ];

    h_pred        = zeros(p, N);
    h_hat_Classic = zeros(p, N);

    h_pred(:,1)        = H(:,1);
    h_hat_Classic(:,1) = H(:,1);

    %--- Kalman loop ---
    for ii = 2:N
        % 1) Predict
        x_pred = A * x_hat(:,ii-1);
        M_pred = A * M * A' + Q;

        % 2) Measurement update
        %    Measurement picks out only the h-portion of the state:
        C = [ V(ii,:), zeros(1,p) ];  % 1×2p
        K = (M_pred * C') / ( rho2 + C*M_pred*C' );

        % 3) Correct
        innov = x(ii) - C * x_pred;
        x_hat(:,ii) = x_pred + K * innov;
        M = ( eye(2*p) - K*C ) * M_pred;

        % 4) Save for output
        h_hat(:,ii)         = x_hat(1:p,ii);
        v_hat(:,ii)         = x_hat(p+1:end,ii);   %# optional if you want v
        h_pred(:,ii)        = x_pred(1:p)';         % the "prior” h
        h_hat_Classic(:,ii) = x_hat(1:p,ii)';       % your filtered h
    end

    MSE_pred(k,:)=mean((abs(H(:,2*p:end-p)-h_pred(:,2*p:end-p)).^2)');
    MSE(k,:)=mean((abs(H(:,2*p:end-p)-h_hat_Classic(:,2*p:end-p)).^2)');

    % MSE(k,:)=mean((abs(H-(h_pred)).^2)');

    %
    % figure;
    % plot(abs(H(1,1:end).^2),'b')
    % hold on
    % plot(abs(h_pred(1,1:end).^2),'r')
    % grid on
    % title(['Tap 1 MSE: ',num2str(10*log10(MSE(1)))]);
    %
    % % % % %
    % figure;
    % plot(real(H(1,2*p:end-p).^2),'b')
    % hold on
    % plot(real(h_pred(1,2*p:end-p).^2),'r')
    % grid on
    % title(['Tap 1 MSE: ',num2str(10*log10(MSE_pred(1)))]);
    %
    % % %
    % figure;
    % plot(abs(H(2,1:end-p).^2),'b')
    % hold on
    % plot(abs(h_pred(2,1:end-p).^2),'r')
    % grid on
    % title(['Tap 2 MSE: ',num2str(10*log10(MSE(2)))]);

    % figure;
    % plot(abs(H(3,1:end-p).^2),'b')
    % hold on
    % plot(abs(h_pred(3,1:end-p).^2),'r')
    % grid on
    % title(['Tap 3 MSE: ',num2str(10*log10(MSE(3)))]);
    %
    % figure;
    % plot(abs(H(4,1:end-p).^2),'b')
    % hold on
    % plot(abs(h_pred(4,1:end-p).^2),'r')
    % grid on
    % title(['Tap 4 MSE: ',num2str(10*log10(MSE(4)))]);
    %
    % figure;
    % plot(abs(H(5,1:end-p).^2),'b')
    % hold on
    % plot(abs(h_pred(5,1:end-p).^2),'r')
    % grid on
    % title(['Tap 5 MSE: ',num2str(10*log10(MSE(5)))]);

    %% Every thing is known - Best possible performance
    % [SER(k),SNRdB_Theory(k),P_err_linear(k)] = Eigen_Beamforming_v1_3(N-(p+1),SNRdB(k), ...
    %     h_0,h_1, h_0,h_1, h_0,h_1);


    %N_BF=19900;
    %% Precoding using Channel estimation no -delay
    % [SER(k),SNRdB_Theory(k),P_err_linear(k)] = Eigen_Beamforming_v1_3(N_BF,SNRdB(k), ...
    %     h_0,h_1,h_0,h_1,h_0,h_1);

    %% Precoding using Channel estimation with -delay using Kalman
    % [SER(k),SNRdB_Theory(k),P_err_linear(k)] = Eigen_Beamforming_v1_3(N-2*p,SNRdB(k), ...
    %     h_0(p:end-p),h_1(p:end-p),h_pred(1,(p:end-p)),h_pred(2,(p:end-p)),h_hat_Classic(1,p:end-p),h_hat_Classic(2,p:end-p));

    % [SER(k),SNRdB_Theory(k),P_err_linear(k)] = Eigen_Beamforming_v1_3(N_BF,SNRdB(k), ...
    %     h_0(p:N_BF+p),h_1(p:N_BF+p),h_pred(1,(1*p:N_BF+1*p)),h_pred(2,(1*p:N_BF+1*p)),h_hat_Classic(1,(1*p:N_BF+1*p)),h_hat_Classic(2,(1*p:N_BF+1*p)));

    % [SER(k),SNRdB_Theory(k),P_err_linear(k)] = Eigen_Beamforming_v1_3(N_BF,SNRdB(k), ...
    %     h_0(1000*p:end),h_1(1000*p:end),h_hat_Classic(1,1:end-1000*p+1),h_hat_Classic(2,1:end-1000*p+1),h_0(1000*p:end),h_1(1000*p:end));


    % [SER(k),SNRdB_Theory(k),P_err_linear(k)] = Eigen_Beamforming_v1_3(N_BF,SNRdB(k), ...
    %     h_0(1*p:end-p),h_1(1*p:end-p),h_pred(1,(1*p:end-p)),h_pred(2,(1*p:end-p)),h_0(1*p:end-p),h_1(1*p:end-p));

    % [SER(k),SNRdB_Theory(k),P_err_linear(k)] = Eigen_Beamforming_v1_3(19580,SNRdB(k), ...
    %     h_0(p:end-p),h_1(p:end-p),h_hat_Classic(1,p+30:end-p-30),h_hat_Classic(2,p+30:end-p-30),h_0(p:end-p),h_1(p:end-p));



    % figure
    % semilogy(SNRdB,SER,SNRdB_Theory,P_err_linear,'-'); grid;
    % xlabel('SNR(dB)');
    % ylabel('SER');
    % title(['Eigen ',num2str(M_MTC),'x ',num2str(N_MRC),' Rayleigh']);
    % legend('Eigen BF 2x2','Theory')

end

% [SNRdB,SER,SNRdB_Theory,P_err_linear] = Eigen_Beamforming_v1_3(SNRdB,h_Quad,h_Quad_1);
% figure
% semilogy(SNRdB,SER,SNRdB_Theory,P_err_linear,'-'); grid;
% xlabel('SNR(dB)');
% ylabel('SER');
% title(['Eigen ',num2str(2),'x ',num2str(2),' Rayleigh']);
% legend('Eigen BF 2x2','Theory')

% figure
% plot(SNRdB,10*log10(MSE));
% hold on
% plot(SNRdB,10*log10(MSE_pred),'*');
% legend('Tap1','Tap2','Tap3','Tap4','Tap5','Tap1 Pred','Tap2 Pred','Tap3 Pred','Tap4 Pred','Tap5 Pred')
% grid on
% xlabel("Rx SNR [dB]")
% ylabel("Channel Prediction MSE")
% title("KF Channel MSE")


end
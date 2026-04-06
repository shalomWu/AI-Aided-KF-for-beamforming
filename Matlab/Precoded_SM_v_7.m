%% Precoded_SM_v_7

clear all;
%close all; clear; clc
tic
set(0, 'DefaultFigureWindowStyle', 'docked');
QPSK_vec=[1+1i, 1-1i, -1+1i, -1-1i]/sqrt(2);
s0_MAP=[ones(1,4)*QPSK_vec(1),ones(1,4)*QPSK_vec(2),ones(1,4)*QPSK_vec(3),ones(1,4)*QPSK_vec(4)];
s1_MAP=[QPSK_vec,QPSK_vec,QPSK_vec,QPSK_vec];
s_MAP=[s0_MAP;s1_MAP];

SNRdB=0;%:5:25;%-10:5:20;%-10:5:20;%-10:5:20;%-10:5:20;
SNRdB_Theory=SNRdB;%0:0.5:20;
SNR=10.^(SNRdB./10);
SNR_Theory=10.^(SNRdB_Theory./10);

%data =
%load("C:\Users\shalomw\Documents\LLM4CP_data\H_true_sub24_Hlong_v3.mat"); % For LLM4CP
data = load("C:\Users\shalomw\Documents\Education\Project ideas\Code\KalmanNet_TSP\Simulations\Linear_canonical\data\Channel_H_4x4.mat"); 
H = data.H;    % replace smyStruct" with the variable name in your file
[N_Rx,M_Tx,N]=size(H);
%H=(ones(N_Rx,M_Tx,N));%+1i*ones(N_Rx,M_Tx,N))/sqrt(2); %% If you want to
%replace  with known channel
%% USe decimation of matrix if required
% NN=input('1 in how many samples to take?: ');
% H_dec = decimate_tensor_3d(H, NN);
% H=H_dec;
% [N_Rx,M_Tx,N]=size(H);

%% Here we load the Knet estimation per SNR
%data1 = load('C:\Users\shalomw\Documents\Education\Project ideas\Code\KalmanNet_TSP\Simulations\Linear_canonical\data\Siamese_13_11_2025\h_Knet_Channel_H_64x6_dec15_SNR+5dB_v148.mat');
%% Insert here the LLM4CP prediction matrix
%data1 = load("C:\Users\shalomw\Documents\LLM4CP_data\H_pred_LLM4CP_sub24_Hlong_v3_SNR+20dB.mat");
%% Insert here the KNet Matrix
data1 = load("C:\Users\shalomw\Documents\Education\Project ideas\Code\KalmanNet_TSP\Simulations\Linear_canonical\data\Siamese_5_4_2026\h_Knet_Channel_H_4x4_SNR+0dB_v402_best_wsr.mat");
Est_or_Pred= input('For [KNetEst, KNetPred, LLM4CP] type [0,1,2]: ');

usePA=input('For passing through PA type 1: ');

if Est_or_Pred==1
    H1 = data1.h_Knet_pred; % replace smyStruct" with the variable name in your file
    %H1(:,:,1)= H(:,:,1);
    H1(:,:,1)= data1.h_Knet(:,:,1);
elseif Est_or_Pred==0
    H1 = data1.h_Knet;    % replace smyStruct" with the variable name in your file
elseif Est_or_Pred==2
    H1= data1.H;
end
%H1 = data1.H;


% N=5285;
% N_Rx=4;
% M_Tx=4;

SER=zeros(N_Rx,length(SNRdB));
EVM=zeros(1,16);
Index_minEVM_ZF=zeros(1,N_Rx);

Decoder_Type = 2;%0=ZF; 1=ML ; 2=MMSE

[NN, MM, LL] = size(H);
LL=input('Do you want to change Length?: ');
H=H(:,:,1:LL);
H1=H1(:,:,1:LL);

SumRate=zeros(length(SNRdB),LL);
SumCapacity=zeros(length(SNRdB),LL);
SumRateSINR=zeros(length(SNRdB),LL);

SumRateAvg=zeros(1,length(SNRdB));
SumCapacityAvg=zeros(1,length(SNRdB));
SumRateSINRAvg=zeros(1,length(SNRdB));



EstimationMethod=input('What you want? Known=0, KF=1, Best Est=2, Knet=3: ');
delay=input('sample delay? [Needed for prediction]: ');
dt=input('What CV do you want: ');
%data = load('Source_4.mat');
%s = data.s;    % replace smyStruct" with the variable name in your file


for k=1:length(SNRdB) %/vector for SNR
    disp('SNR_dB:')
    disp(SNRdB(k));
    s_hat=zeros(N_Rx,LL);
    rho=1./sqrt(SNR(k));
    y=zeros(N_Rx,LL);
    NMSE=zeros(N_Rx,LL);

    %% Signal Generation
    s=randsrc(N_Rx,LL,QPSK_vec);
    % s(1,:)=randsrc(1,N,QPSK_vec);
    % s(2,:)=s(1,:);


    %% Taking each Tx-Rx channel and finding it's KF estimation
    % Assuming H is of size [N, M, L]
    %
    % Preallocate normalized tensor
    h_KF = zeros(NN, MM, LL);
    h_hat_Classic=zeros(NN, MM, LL);
    H_norm=zeros(NN, MM, LL);
    H_PA=zeros(NN, MM, LL);
    h_hat_Knet=zeros(NN, MM, LL);
    %
    % Loop through each channel and normalize
    for n = 1:NN
        for m = 1:MM
            c = squeeze(H(n, m, :))';       % Extract Lx1 vector
            %c = H(n, m, :);
            %norm_val = norm(h, 2);         % Compute 2-norm
            norm_val=1/sqrt(LL/sum(abs(c).^2));
            if norm_val ~= 0
                H_norm(n, m, :) = c / norm_val;
                %% Here we need to add the compression stage of the PA
                if usePA==1

                    % V1=V.*exp(1i*0.1*(rand(1,length(V))-0.5)*2*pi)';
                    %V1=V.*exp(1i*0.1*(rand(1,length(V))-0.0)*2*pi)';
                    %% Passing the channel through PA
                    ampmParams.Phi_max_deg = 40+5*randn;   % max phase shift at saturation
                    ampmParams.order       = 3;   % linear vs normalized amplitude
                    [G, ~, ~] = pa_model(c / norm_val, 'SSPA', 1+0.2*randn, 5, 'poly', ampmParams);

                else
                    G = c / norm_val; %H_norm(n, m, :);
                end
                H_PA(n, m, :) = G;

                %% End of PA model
                %[h_KF(n, m, :), h_hat_Classic(n, m, :)] = Kay_Example13_3_v_3(N,SNRdB(k), c/norm_val,0.98,0.045,100); % Mke sure that if ver1 is used the sizes must be adapted
                [h_KF(n, m, :), h_hat_Classic(n, m, :)] = Kay_Example13_3_v_4(LL,SNRdB(k),G,0.98,0.01,1,0.04,dt); %0.98,0.01,1,0.04%1,0.1,1,0.8,0.3 % Mke sure that if ver1 is used the sizes must be adapted
                h_hat_Knet(n,m,:)=squeeze(H1(n, m, :))'/(1/sqrt(LL/sum(abs(squeeze(H1(n, m, :))).^2)));
                %h_hat_Knet(n,m,:)=squeeze(H1(n, m, :))'/(1/sqrt(LL/sum(abs(squeeze(H(n, m, :))).^2))); %Dont use this Need to try this for low SNR

            end
        end
    end


    %% Sanity Check to see link behaviour
    % figure
    % plot(imag(squeeze((h_hat_Knet(1,1,delay:end)))),'r');
    % %plot(imag(squeeze((h_hat_Knet(1,1,delay:end)))),'r');
    % hold on
    % plot(imag(squeeze(h_hat_Classic(1,1,1:end-delay))),'b');
    % hold on
    % plot(imag(squeeze(h_KF(1,1,1:end-delay))),'b--');
    % hold on
    % plot(imag(squeeze(H_norm(1,1,1:end-delay))),'black');
    % hold on
    % plot(imag(squeeze(H_PA(1,1,1:end-delay))),'g');
    % grid on
    % legend('Knet','Est MB KF','Pred MB KF','Full knowledge',"Channel Compressed")
    % %%
    % figure
    % plot(abs(squeeze((h_hat_Classic(1,1,delay:end)))),'b');
    % hold on;
    %  plot(abs(squeeze((h_hat_Knet(1,1,delay:end)))),'r');
    % hold on;
    % plot(abs(squeeze((H_norm(1,1,delay:end)))),'black');
    % hold on;
    % grid on
    % legend('MB KF','Knet','Full knowledge')
    % %
    % figure
    % plot(phase(squeeze((h_hat_Classic(1,1,delay:end)))),'b');
    % hold on;
    % plot(phase(squeeze((h_hat_Knet(1,1,delay:end)))),'r');
    % hold on;
    % plot(phase(squeeze((H_norm(1,1,delay:end)))),'black');
    % hold on;
    %   legend('MB KF','Knet','Full knowledge')
    % grid on


    % %% Graphs with no delays. PLot these  March 2026
    % figure
    % plot(imag(squeeze((h_hat_Knet(1,1,:)))),'r');
    % %plot(imag(squeeze((h_hat_Knet(1,1,delay:end)))),'r');
    % hold on
    % plot(imag(squeeze(h_hat_Classic(1,1,:))),'b');
    % hold on
    % plot(imag(squeeze(h_KF(1,1,:))),'b--');
    % hold on
    % plot(imag(squeeze(H_norm(1,1,:))),'black');
    % hold on
    % plot(imag(squeeze(H_PA(1,1,:))),'g');
    % grid on
    % legend('Knet','Est MB KF','Pred MB KF','Full knowledge',"Channel Compressed")
    % %%
    % figure
    % plot(abs(squeeze((h_KF(1,1,:)))),'b');
    % hold on;
    %  plot(abs(squeeze((h_hat_Knet(1,1,:)))),'r');
    % hold on;
    % plot(abs(squeeze((H_norm(1,1,:)))),'black');
    % hold on;
    % grid on
    % legend('MB KF','Knet','Full knowledge')
    % %
    % figure
    % plot(phase(squeeze((h_KF(1,1,:)))),'b');
    % hold on;
    % plot(phase(squeeze((h_hat_Knet(1,1,:)))),'r');
    % hold on;
    % plot(phase(squeeze((H_norm(1,1,:)))),'black');
    % hold on;
    % legend('MB KF','Knet','Full knowledge')
    % grid on
    % %% The 2 graphs above are to plot

    %% Here is the place to change the channel for sanity
    %h=H_norm;%/norm(H,'fro'); % QuaDRigA realizations
    h=H_PA; % If we want to combine H and G

    %h=(ones(N_Rx,M_Tx,N)+1i*ones(N_Rx,M_Tx,N))/sqrt(2); %Known channel
    %h=(randn(N_Rx,M_Tx,N)+1i*randn(N_Rx,M_Tx,N))/sqrt(2); %Rayliegh - Amplitude and Phase change of all the Multipaths per symbol

    n=(randn(N_Rx,LL)+1i*randn(N_Rx,LL))/sqrt(2); %AWGN
    %% End of Channel setting
    SINR = zeros(LL,N_Rx);
    for ii=1:LL-(delay-1)
        %h(:,:,ii)=[1.5,1;1.5,1]; %Sanity check to make sure we have (2^2)^2=16 options

        %% Using eigenvalues and vectors

        %h(:,:,ii)=eye(N_Rx); % This is for sanity check. N_Rx orthonormal streams independent of each other

        if EstimationMethod==0
            [U,S,V]=svd(h(:,:,ii)); %This is for the real h
            h_MUI=h(:,:,ii);
        elseif EstimationMethod==1
            [U,S,V]=svd(h_KF(:,:,ii)); %This is using the prediction
            h_MUI=h_KF(:,:,ii);
        elseif EstimationMethod==2
            [U,S,V]=svd(h_hat_Classic(:,:,ii)); %This is using the prediction
            h_MUI=h_hat_Classic(:,:,ii);
        elseif EstimationMethod==3
            [U,S,V]=svd(h_hat_Knet(:,:,ii+delay-1)); %This is using the prediction
            h_MUI=h_hat_Knet(:,:,ii+delay-1);
            %[U,S,V]=svd(conj(h_hat_Knet(:,:,ii+delay-1))); %This is using the prediction
            %h_MUI=conj(h_hat_Knet(:,:,ii+delay-1));
        else
            %[U,S,V]=svd(h(:,:,ii));
            h_MUI=h(:,:,ii);
        end

        %% Predcoder for SU-MIMO w is the SVD precoder that allows Tx orthogonality over the H channel
        %w=V;

        %% Basic ZF (one shot) Predcoder for MU-MIMO
        % D=eye(N_Rx);
        % w0=pinv(h_MUI)*D/norm(pinv(h_MUI)*D); %
        %w0=randn(M_Tx,N_Rx);
        w0=V;
        %w0=ones(size(V));
        %[p_i, mu] = wf_power_alloc(S, 1, rho);

        %% Use PGA algorithm

        % w = pga_precoder(h_MUI, 1, rho^2, 6, w);
        % w=sqrt(N_Rx)*w;

        % pga_precoder_line_search(h_MUI, 1, rho^2, 10, w);
        % w=sqrt(N_Rx)*w;

        %% Use WMMSE algorithm
        %w = wmmse_precoder(h_MUI, 1, rho^2, 15);
        %w = wmmse_precoder_with_lambda(h_MUI, 1, rho^2, 15, w0);
        w = wmmse_precoder_with_lambda_v1(h_MUI, 1, rho^2, 15, w0);
        %w=  wmmse_precoder_with_lambda_SU_MIMO(h_MUI, 1, rho^2, 100, w0);
        %w=w/norm(w,'fro');
        w=w/norm(w);
        %w=w*sqrt(N_Rx);
        %w=ones(size(w));
        %% Signal Generation
        %y(:,ii) =1/sqrt(N_Rx)*h(:,:,ii)*(w(:,1:N_Rx)*s(:,ii))+rho*n(:,ii);% in case 2x2
        y(:,ii) =1/sqrt(M_Tx)*h(:,:,ii)*(w(:,1:N_Rx)*s(:,ii))+rho*n(:,ii);% in case 2x2


        %y(:,ii) =sqrt(fliplr(p_i')).*h(:,:,ii)*(w(:,1:N_Rx)*s(:,ii))+rho*n(:,ii);% in case 2x2
        %y(:,ii) =sqrt([0.5,0.5,0,0]').*h(:,:,ii)*(w(:,1:N_Rx)*s(:,ii))+rho*n(:,ii);% in case 2x2

        %% SumRate Calculation
        %SumRate(k,ii)=log2(det(eye(N_Rx)+(SNR(k)/N_Rx)*h(:,:,ii)*w(:,1:N_Rx)*w(:,1:N_Rx)'*h(:,:,ii)'));
        SumRate(k,ii)=log2(det(eye(N_Rx)+(SNR(k)/M_Tx)*h(:,:,ii)*w(:,1:N_Rx)*w(:,1:N_Rx)'*h(:,:,ii)'));

        %SumCapacity(k,ii)=sum(log2(1+diag(S)*SNR(k)/2/N_Rx));

        %M=U'*(h(:,:,ii)*w); % This should be used for SU-MIMO
        M=(h(:,:,ii)*w);

        SINR = zeros(1,N_Rx);
        for i = 1:N_Rx
            signal_power     = abs(M(i,i))^2;
            interf_power     = sum(abs(M(i,setdiff(1:N_Rx,i))).^2);
            noise_power      = rho^2;
            SINR(ii,i)          = signal_power / (interf_power + noise_power);
        end
        SumRateSINR(k,ii)=sum(log2(1+SINR(ii,:)/M_Tx));
        %SumRateSINR(k,ii)=sum(log2(1+SINR(ii,:)/N_Rx));

        clear i;

        %% ZF Slicer


        if Decoder_Type==0 %ZF decoder when all H is known for every user - Good for SU-MIMO
            %s_hat(:,ii)=pinv(1/sqrt(N_Rx)*h(:,:,ii)*w(:,1:N_Rx))*y(:,ii); %This is with perfect knowledge of h
            s_hat(:,ii)=pinv(1/sqrt(M_Tx)*h(:,:,ii)*w(:,1:N_Rx))*y(:,ii); %This is with perfect knowledge of h

            %s_hat(:,ii)=(w(:,1:N_Rx)*h(:,:,ii))^-1*y(:,ii);
            %s_hat(:,ii)=h(:,:,ii)^-1*y(:,ii);
            %s_hat(:,ii)=pinv(h(:,:,ii)).*w*y(:,:);
            %s_hat(:,ii)=(h(:,:,ii)*w)'*y(:,:);
            %s_hat(:,ii)=pinv(1/sqrt(M_Tx)*h_hat_Classic(:,:,ii)*w(:,1:N_Rx))*y(:,ii); %This is with best estimation of h
            % s_hat(:,ii)=pinv(1/sqrt(M_Tx)*h(:,:,ii)*w)*y(:,:);
            %s_hat(:,ii)=y(:,:)*pinv(1/sqrt(M_Tx)*h(:,:,ii)*w);
            %s_hat(:,ii)=y(1);
        elseif Decoder_Type==1 % ML decoder - Relevant only for 2x2 case
            for jj=1:16
                EVM(jj)=norm(y-y_MAP(:,jj));
            end
            Index_minEVM = find(EVM==min(EVM));
            s_hat(:,ii)= s_MAP(:,Index_minEVM(1));

        elseif Decoder_Type==2 % NU-MIMO MMSE decoder when precoding is known for all user but not the channel
            %[s_hat(:,ii), ~] = mmse_decoder(h(:,:,ii), w', y(:,ii), rho^2);
            [s_hat(:,ii), ~] = mmse_decoder_v1(h(:,:,ii)', w, y(:,ii), rho^2);
        end

    end

    %% Error Calculation of SER

    SumRateAvg(k)=mean(abs(SumRate(k,10:end)));
    %SumCapacityAvg(k)=mean(SumCapacity(k,:));
    SumRateSINRAvg(k)=mean(SumRateSINR(k,10:end));

    %% PLoting the CDF and histogram of the rate
    % figure;
    % cdfplot(SumRateSINR(k,:));
    % hold on
    % cdfplot(abs(SumRate(k,:)));
    %
    % grid on;
    % xlabel('x');
    % ylabel('Empirical CDF');
    % title('Empirical CDF via cdfplot');
    %
    % figure
    % hist(SumRateSINR(k,:),1000);
    % grid on
    % hold on
    % hist(abs(SumRate(k,:)),1000);
    %
    %% plot the Rate. Uncomment here
    % figure
    % plot(SumRateSINR(k,:),'b');
    % hold on
    % plot(abs(SumRate(k,:)),'r')
    % grid on

    %% Rnd of plots
    s_tilde=sign(real(s_hat))/sqrt(2)+1i*sign(imag(s_hat))/sqrt(2);
    NumOfErrors=sum(abs(s_tilde-s)>eps,2);
    SER(:,k)=NumOfErrors/(LL);

    %         figure;
    %         plot(NMSE(1,:),'b.')
    %         hold on
    %         plot(NMSE(2,:),'r.')
    %         title(['NMSE vs. time, SNR: ',num2str(SNRdB(k))])
    %         ylabel('NMSE [dB]')
end

%% Theoritcal BER Curve for SM
P_err_linear=1./(1+((N_Rx/M_Tx)*SNR_Theory)/(2*N_Rx)).^N_Rx; %SM BER Curve
%P_err_linear=1./(1+((1/M_Tx)*SNR_Theory)/(2*1)).^1; %SM 2x1 specific \ZF

%% Plotting the BER Curves
% figure
% semilogy(SNRdB,SER,'x',SNRdB_Theory,P_err_linear,'black-'); grid;
% xlabel('SNR (dB)');
% ylabel('SER of each stream');



R=0;
for jj=1:N_Rx
    R = R+ SERtoRate(SER(jj,:));
end

figure;
plot(SNRdB, R, 'b-', 'LineWidth', 1); hold on;
%plot(SNRdB, N_Rx*log2(1 + SNR/2/N_Rx), 'k--', 'LineWidth', 0.5);
%plot(SNRdB, SumCapacityAvg, 'g-', 'LineWidth', 1);
plot(SNRdB, SumRateAvg, 'm--', 'LineWidth', 1);
plot(SNRdB, SumRateSINRAvg, 'k-', 'LineWidth', 1);

grid on;
xlabel('SNR (dB)');
xticks(min(SNRdB):1:max(SNRdB));
ylabel('Achievable Rate (bits/sec/Hz)');
title(['MU MIMO ',num2str(M_Tx),' x ',num2str(N_Rx),' Capacity Estimate ']);
%legend('Estimated from SER', 'SISO AWGN Capacity','Sum Capacity','SumRate Equation','SINR based', 'Location', 'NorthWest');
legend('Estimated from SER','SumRate Equation','SINR based', 'Location', 'NorthWest');

% SumRateAvg
SumRateSINRAvg
%% save the PA file to be trained on later
if usePA==1
    H = conj(H_PA);
    save('C:\Users\shalomw\Documents\Education\Project ideas\Code\KalmanNet_TSP\Simulations\Linear_canonical\data\Channel_H_4x4_v4_PA_v1.mat', 'H');
end
%%
toc
% figure;
% plot(SNRdB, 100*SumRateAvg, 'm--', 'LineWidth', 1);
%
% grid on;
% xlabel('SNR (dB)');
% xticks(min(SNRdB):1:max(SNRdB));
% ylabel('Achievable Rate (bits/sec/Hz)');
% title('Sumrate equation');
% legend('SumRate Equation', 'Location', 'NorthWest');







%% Backup
%% Sanity check plotting the constellation
% figure; plot(y_MAP(1,:),'r*')
% hold on; plot(y(1,:),'b.')
% grid on
% title(' Recieved constealltion chain 1')
%
% figure; plot(y_MAP(2,:),'r*')
% hold on; plot(y(2,:),'b.')
% grid on
% title(' Recieved constealltion chain 2')

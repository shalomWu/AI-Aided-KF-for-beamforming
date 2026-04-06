clear
data = load('C:\Users\shalomw\Documents\LLM4CP_data\H_pred_LLM4CP_sub24_Hlong_v3.mat');
H = data.H;    % replace smyStruct" with the variable name in your file
[N_Rx,M_Tx,N]=size(H);
LL=N;

SNRdB=-10:5:25;
% c=c;
% c1=c1;
%N=5285;
M=1;
p =1;


MSE_pred=zeros(1,length(SNRdB));
MSE=zeros(1,length(SNRdB));

for m=1:M_Tx
    for n=1:N_Rx
        disp('m,n:')
        disp(m);disp(n);
        c = squeeze(H(1, 1, 1:LL))'; %*sqrt(N/sum(abs(H(1, 1, :)).^2));
        c1 = squeeze(H(n, m, 1:LL))';
        % c=H(1, 1, :);%*sqrt(N/sum(abs(H(1, 1, :)).^2));
        % c1=H(2, 1, :);




        Quad=(c*sqrt(LL/sum(abs(c).^2)));
        Quad_1=(c1*sqrt(LL/sum(abs(c1).^2)));
        % Quad=(randn(1,N)+1i*randn(1,N))/sqrt(2);%(ones(1,N)+1i*ones(1,N))/sqrt(2);%
        % Quad_1=(randn(1,N)+1i*randn(1,N))/sqrt(2);%(ones(1,N)+1i*ones(1,N))/sqrt(2);%

        % c_rep=repelem(c,M);
        % c1_rep=repelem(c1,M);
        % c_rep=(c_rep*sqrt(N*M/sum(abs(c_rep).^2)));
        % c1_rep=(c1_rep*sqrt(N*M/sum(abs(c1_rep).^2)));


        SER=zeros(1,length(SNRdB));
        SNRdB_Theory=zeros(1,length(SNRdB));
        P_err_linear=zeros(1,length(SNRdB));

        MSE_pred_1=zeros(1,length(SNRdB));
        MSE_1=zeros(1,length(SNRdB));


        for k=1:length(SNRdB)
            disp(['SNR: ', num2str(SNRdB(k)),'dB']);
            %[h_pred, h_hat_Classic] = Kay_Example13_3_v_5(N,SNRdB(k),Quad_1,0.92,0.05,1);
            %[h_pred, h_hat_Classic] = Kay_Example13_3_v_4(N,SNRdB(k),Quad_1,0.98,0.01,1,0.04);
            [h_pred, h_hat_Classic] = Kay_Example13_3_v_4(LL,SNRdB(k),Quad_1,0.98,0.01,1,0.04,0.09);
            %[h_pred, h_hat_Classic] = Kay_Example13_3_v_4(N,SNRdB(k),Quad_1,1,0.1,1,0.8,0.4);

            h_pred=h_pred*sqrt(LL/sum(abs(h_pred).^2));
            h_hat_Classic=h_hat_Classic*sqrt(LL/sum(abs(h_hat_Classic).^2));

            %[h_pred, h_hat_Classic] = Kay_Example13_3_v_3(N,SNRdB(k),Quad_1,0.98,0.04,100);
            %[h_pred, h_hat_Classic] = Kay_Example13_3_v_2(N,SNRdB(k),Quad_1);

            % MSE_pred(k)=mean((abs(Quad(2*p:end-p)-h_pred(1,2*p:end-p)).^2)./abs(Quad(2*p:end-p)).^2);
            % MSE(k)=mean((abs(Quad(2*p:end-p)-h_hat_Classic(1,2*p:end-p)).^2)./abs(Quad(2*p:end-p)).^2);
            %% This is what commented
            % MSE_pred(k)=mean((abs(Quad_1-       h_pred).^2)./abs(Quad_1).^2);
            % MSE(k)=     mean((abs(Quad_1-h_hat_Classic).^2)./abs(Quad_1).^2);
            %%
            MSE_pred(k)=MSE_pred(k)+mean((abs(Quad_1-       h_pred).^2)./abs(Quad_1).^2);
            MSE(k)=  MSE(k)+   mean((abs(Quad_1-h_hat_Classic).^2)./abs(Quad_1).^2);



            %MSE_pred_1(k)=mean((abs(Quad_1(2*p:end-p)-h_pred_1(1,2*p:end-p)).^2)./abs(Quad_1(2*p:end-p)).^2);
            %MSE_1(k)=mean((abs(Quad_1(2*p:end-p)-h_hat_Classic_1(1,2*p:end-p)).^2)./abs(Quad_1(2*p:end-p)).^2);



            %h_pred_repl=repelem(h_pred(1,:), M);
            %h_pred_1_repl=repelem(h_pred_1(1,:), M);

            % h_hat_Classic_repl=repelem(h_hat_Classic(1,:), M);
            %h_hat_Classic_1_repl=repelem(h_hat_Classic_1(1,:), M);


            % [SER(k),SNRdB_Theory(k),P_err_linear(k)] = Eigen_Beamforming_v1_3(N*M-2000,SNRdB(k), ...
            %     c_rep(1000:end-1000),c1_rep(1000:end-1000),h_hat_Classic_repl(1:end),h_hat_Classic_1_repl(1:end),h_hat_Classic_repl(1000:end-1000),h_hat_Classic_1_repl(1000:end-1000));

            % [SER(k),SNRdB_Theory(k),P_err_linear(k)] = Eigen_Beamforming_v1_3(N*M,SNRdB(k), ...
            %     c_rep,c1_rep, h_hat_Classic_repl,h_hat_Classic_1_repl,c_rep,c1_rep);


            % [SER(k),SNRdB_Theory(k),P_err_linear(k)] = Eigen_Beamforming_v1_3(N*M,SNRdB(k), ...
            %     flip(c_rep), flip(c1_rep), flip(c_rep), flip(c1_rep), flip(c_rep), flip(c1_rep));
            %

            % [SER(k),SNRdB_Theory(k),P_err_linear(k)] = Eigen_Beamforming_v1_3(N,SNRdB(k), ...
            %     Quad,Quad_1,Quad,Quad_1,Quad,Quad_1);
        end


        %
        % figure
        % plot(real(Quad_1),'k');hold on,plot(real(h_pred),'r')
        % grid on
        % title("Prediction")
        %
        % figure
        % plot(real(Quad_1),'r');hold on,plot(real(h_hat_Classic),'k')
        % grid on
        % title("Estimation")
        % xlabel(' sample number')
        % ylabel('Real value of H')

        % figure
        % plot(real(c1_rep),'b');hold on,plot(real(h_pred_1_repl),'r')
        % grid on

        % figure
        % plot(real(c_rep),'b');hold on,plot(real(h_hat_Classic_repl),'r')
        % grid on

        %
        % figure
        % semilogy(SNRdB,SER,SNRdB_Theory,P_err_linear,'-'); grid;
        % xlabel('SNR(dB)');
        % ylabel('SER');
        % title(['Eigen ',num2str(2),'x ',num2str(2),' Rayleigh']);
        % legend('Eigen BF 2x2','Theory'b


        % figure
        % hold on
        % plot(SNRdB,10*log10(MSE),'k');%,SNRdB,10*log10(MSE_1),'b');
        % hold on
        % plot(SNRdB,10*log10(MSE_pred),'k*');%,SNRdB,10*log10(MSE_pred_1),'b*');
        % legend('Estimation Ch0','Prediction Ch0');%,'Prediction Ch0','Prediction Ch1')
        % grid on
        % xlabel("Rx SNR [dB]")
        % ylabel("Channel Prediction NMSE")
        % title("KF Channel NMSE")


    end
end

figure
hold on
plot(SNRdB,10*log10(MSE/(M_Tx*N_Rx)),'k');%,SNRdB,10*log10(MSE_1),'b');
hold on
plot(SNRdB,10*log10(MSE_pred/(M_Tx*N_Rx)),'b*');%,SNRdB,10*log10(MSE_pred_1),'b*');
legend('Estimation Ch0','Prediction Ch0');%,'Prediction Ch0','Prediction Ch1')
grid on
xlabel("Rx SNR [dB]")
ylabel("Channel Prediction NMSE")
title("KF Channel NMSE")

% figure
% hold on
% plot(SNRdB,-10*log10(var(MSE/(M_Tx*N_Rx))),'k');%,SNRdB,10*log10(MSE_1),'b');
% hold on
% plot(SNRdB,-10*log10(var(MSE_pred/(M_Tx*N_Rx))),'b*');%,SNRdB,10*log10(MSE_pred_1),'b*');
% legend('Estimation Ch0','Prediction Ch0');%,'Prediction Ch0','Prediction Ch1')
% grid on
% xlabel("Rx SNR [dB]")
% ylabel("Channel Prediction NMSE")
% title("KF Channel NMSE")
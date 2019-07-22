classdef fftLayer < nnet.layer.Layer
    %   Detailed explanation goes here
    properties (Learnable)
        
    end
    
    methods
        function layer = fftLayer(name) 
            % Set layer name
            if nargin > 1
                layer.Name = name;
            end
            % Set layer description
            layer.Description = 'fftLayer'; 
        end
        function Z = predict(~,x)
            % Forward input data through the layer and output the result
            Z = fft(x);
            %Z
            
        end
        function dLdX = backward(layer,x,Z,dLdZ,~)
            % Backward propagate the derivative of the loss function through 
            % the layer 
            % dLdX = X.*(1-X) .* dLdZ; % original code
            dLdX =Z.*(1-Z).*dLdZ;
        end
        
    end
end



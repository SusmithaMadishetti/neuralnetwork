classdef ifftLayer < nnet.layer.Layer
    
    %   Detailed explanation goes here
    
    properties (Learnable)
       
    end
    
    methods
        function layer = ifftLayer(name) 
            % Set layer name
            if nargin > 1
                layer.Name = name;
            end
            % Set layer description
            layer.Description = 'ifftLayer'; 
        end
        function Z = predict(~,x)
           % x
            % Forward input data through the layer and output the result
            Z = ifft(x,'symmetric');
            
        end
        function dLdX = backward(layer,x,Z,dLdZ,~)
            % Backward propagate the derivative of the loss function through 
            % the layer 
            % dLdX = X.*(1-X) .* dLdZ; % original code
            dLdX =Z.*(1-Z).*dLdZ.* (Z >= 0);
        end
        
     end
end


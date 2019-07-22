% converts a 1d input column vector to 2d matrix 
classdef convert1d2dLayer < nnet.layer.Layer
    methods
        function layer = sigmoidLayer(name) 
            % Set layer name
            if nargin == 2
                layer.Name = name;
            end
            % Set layer description
            layer.Description = 'convert 1d input to 2d output'; 
        end
        
        function Z = predict(~,X)
            % Forward input data through the layer and output the result
            s2 = size(X, 3); % get number of elements
            s = sqrt(s2);
            
            % get all channels into one vector
            Xt = reshape(X, numel(X)/size(X,4), 1, 1, size(X, 4));
            
            % preallocate array
            if isa(Xt, 'gpuArray')
                % necessary if input is of type gpuArray
                Z = zeros(s, s, size(Xt, 3),size(Xt, 4),...
                    classUnderlying(Xt), class(Xt));
            else
                Z = zeros(s, s, size(Xt, 3),size(Xt, 4), class(Xt));
            end
            
            
            for a = 1:size(Xt, 4)
                for b = 1:s
                    for c = 1:s
                        Z(c, b, :,a) = ...
                            Xt((b-1)*s+c, 1, 1,a);
                    end
                end
            end
        end
        function dLdX = backward(~,X, ~,dLdZ,~)
            % Backward propagate the derivative of the loss function through 
            % the layer 
            
            s2 = numel(X)/size(X,4); % get number of elements
            s = sqrt(s2);
            
            % if input is of size (1,1,s2,2), do this
            if size(X,3) > size(X,1)
                % preallocate array
                if isa(X, 'gpuArray')
                    % necessary if input is of type gpuArray
                    dLdX = zeros(1, 1, s2,size(dLdZ, 4),...
                        classUnderlying(X), class(X));
                else
                    dLdX = zeros(1, 1, s2,size(dLdZ, 4), class(X));
                end

                for a = 1:size(dLdZ,4)
                    for b = 1:s
                        for c = 1:s


                            if ((size(dLdZ,1) >= c) && (size(dLdZ,2) >= b))
                                dLdX(1, 1, (b-1)*s+c,a) = ...
                                    dLdZ(c,b,1,a);
                            end

                        end
                    end
                end
            
            else
            
                % preallocate array
                if isa(X, 'gpuArray')
                    % necessary if input is of type gpuArray
                    dLdX = zeros(s2, 1, 1,size(dLdZ, 4),...
                        classUnderlying(X), class(X));
                else
                    dLdX = zeros(s2, 1, 1,size(dLdZ, 4), class(X));
                end

                for a = 1:size(dLdZ,4)
                    for b = 1:s
                        for c = 1:s


                            if ((size(dLdZ,1) >= c) && (size(dLdZ,2) >= b))
                                dLdX((b-1)*s+c, 1, 1,a) = ...
                                    dLdZ(c,b,1,a);
                            end


                        end
                    end
                end
            end
        end
    end
end
classdef ifftLayer < nnet.layer.Layer
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access=public)
        Property1 
    end
    
    methods
        function obj = ifftLayer(input)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            obj.Property1 = ifft(input);
        end
        
     end
end


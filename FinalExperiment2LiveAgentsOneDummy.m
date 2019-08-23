%set values
NoTrials = 150;
T = 25;
dt = 0.05;
time = 0:dt:T;
noWeights=9;
N=3;


agent = bestIndividual;
agentWeightsPacked = agent(:,1:noWeights);
agentWeights = formMatrix(agentWeightsPacked,N);
sensorGain = agent(:,noWeights+1);
outputGain =agent(:,noWeights+2);
outputGain2 = agent(:,noWeights+3);
agent2Locations = [];
agent1Locations=[];
agent3Locations=[];
agent2InitActiv =[];
allFitnessScoresOneDummy2Live=[];
dummyAgent = recordedAgent;
fitnessScoresOneDummy2Live = [];
    
for t = 1:NoTrials
    initialConditions = randn(N,1);
    initialConditions2 = randn(N,1);
    initialConditions3 = randn(N,1);
    noiseSD = sqrt(0.5);
    noiseMean = 0;
    noise = noiseSD.*randn(1,length(time)) + noiseMean;
    noise2 = noiseSD.*randn(1,length(time)) + noiseMean;
    noise3 = noiseSD.*randn(1,length(time)) + noiseMean;
    noise4 = noiseSD.*randn(1,length(time)) + noiseMean;
    noise5 = noiseSD.*randn(1,length(time)) + noiseMean;
    noise6 = noiseSD.*randn(1,length(time)) + noiseMean;
    bla = linspace(-200,0,1000);
    %set agents up
    %starting conditions - agent one
    agentOne = zeros(N,length(time));
    %agent initial conditions
    agentOne(:,1) = initialConditions(:,:);
     
    %agent Three
    agentThree = zeros(N,length(time));
    agentThree(:,1) = initialConditions3(:,:);

    bla = linspace(-200,0,1000);

    
    %set start locations
    agent1StartPoint =0;
    agentOneLocation = zeros(1,length(time));
    agentOneLocation(:,1) = agent1StartPoint;
    
    agentThreeLocation = zeros(1,length(time));
    %agent3StartPoint = startPoints2(:,p);
    agentThreeLocation(:,1) = agent1StartPoint;%starts with agent one
   %agent  input
    agentOneI = zeros(N,1);
    
    agentThreeI = zeros(N,1); 

    %array storing locations where the agents cross
    crossLocations = [];

    for i= 2:length(time)
            
            if(rand<0.3)
                agentOneI(:,:)=0;
                
                agentThreeI(:,:)=0;
            end
            

            %integrated equation of CTRNN agent one
            agentOne(:,i) = agentOne(:,i-1) +dt*(-agentOne(:,i-1)+tanh(agentWeights*agentOne(:,i-1)+ agentOneI(:,:)));
            
            %integrated equation of CTRNN agent three
            agentThree(:,i) = agentThree(:,i-1) +dt*(-agentThree(:,i-1)+tanh(agentWeights*agentThree(:,i-1)+ agentThreeI(:,:)));
            
            %agent Velocity
            
            agentOneVelocityLeft =(agentOne(2,i)+noise3(:,i))*outputGain;
            agentOneVelocityRight = (agentOne(3,i)+noise4(:,i))*outputGain2;
            agentOneVelocity = agentOneVelocityLeft-agentOneVelocityRight;
            agentThreeVelocityLeft = (agentThree(2,i) + noise5(:,i))*outputGain;
            agentThreeVelocityRight = (agentThree(3,i) + noise6(:,i))*outputGain;
            agentThreeVelocity = (agentThreeVelocityLeft-agentThreeVelocityRight);

            %agent location
           
            agentOneLocation(:,i) = agentOneLocation(:,i-1) - (agentOneVelocity);
            agentThreeLocation(:,i) = agentThreeLocation(:,i-1)+(agentThreeVelocity);

            %input is distance to other agent mapped between 1 and 0 only
            %when agents are within 0 - 200 units of space to eachother
            %on-off sensing essentially
            distanceBetweenAgentOneTwo = -(abs(dummyAgent(:,i) - agentOneLocation(:,i)));
            distanceBetweenAgentOneThree = -(abs(agentThreeLocation(:,i) - agentOneLocation(:,i)));
            distanceBetweenAgentTwoThree = -(abs(agentThreeLocation(:,i) - dummyAgent(:,i)));
            if(distanceBetweenAgentOneTwo > -200 && distanceBetweenAgentOneTwo< 0)
                bla =[bla distanceBetweenAgentOneTwo];
                norm_data = (bla - min(bla)) / ( max(bla) - min(bla) );
                agentOneI(1,:) = norm_data(end)*sensorGain;
                
            else
                agentOneI(1,:) = 0;
               
            end
            if(distanceBetweenAgentOneThree > -200 && distanceBetweenAgentOneThree< 0)
                bla =[bla distanceBetweenAgentOneThree];
                norm_data = (bla - min(bla)) / ( max(bla) - min(bla) );
                agentOneI(2,:) = norm_data(end)*sensorGain;
                agentThreeI(1,:)= norm_data(end)*sensorGain;
            else
                agentOneI(2,:) = 0;
                agentThreeI(1,:)=0;
            end
            if(distanceBetweenAgentTwoThree > -200 && distanceBetweenAgentTwoThree< 0)
                bla =[bla distanceBetweenAgentTwoThree];
                norm_data = (bla - min(bla)) / ( max(bla) - min(bla) );
                
                agentThreeI(2,:)= norm_data(end)*sensorGain;
            else
               
                agentThreeI(2,:)=0;
            end
            
            agentLocations = [agentOneLocation(:,i),dummyAgent(:,i),agentThreeLocation(:,i)];
            if(range(agentLocations)<40)
                crossLocations = [crossLocations agentOneLocation(:,i)];
            end

         
         
            
           

            
    end

    if(isempty(crossLocations))
        fitness= 0;
    else
        fitness = abs(crossLocations(end));

    end
    fitnessScoresOneDummy2Live(t)= fitness;
    allFitnessScoresOneDummy2Live=[allFitnessScoresOneDummy2Live fitness];
    agent1LocationsOneDummy2Live=[agent1LocationsOneDummy2Live; agentOneLocation];
    
    agent3LocationsOneDummy2Live = [agent3LocationsOneDummy2Live;agentThreeLocation];
    
end


meanFitnessOneDummy2Live = mean(fitnessScoresOneDummy2Live);
medianFitnessOneDummy2Live = median(fitnessScoresOneDummy2Live);
stdDevOneDummy2Live = std(fitnessScoresOneDummy2Live);


%{
TF = isoutlier(fitnessScoresOneDummy2Live);
count = length(fitnessScoresOneDummy2Live);
while(count>0)
    
    if(TF(:,count) == 1)
        fitnessScoresOneDummy2Live(:,count) = [];
        
        count =count- 1;
    else
        count =count- 1;
    end
end


meanFitnessTwoLiveOutliersRemoved = mean(fitnessScoresOneDummy2Live);
medianFitnessTwoLiveOutliersRemoved = median(fitnessScoresOneDummy2Live);
stdDevTwoLiveOutliersRemoved = std(fitnessScoresOneDummy2Live);
iqrangeTwoLive = iqr(fitnessScoresOneDummy2Live);
stdErrTwoLive = std(fitnessScoresOneDummy2Live)/sqrt(length(fitnessScoresOneDummy2Live));

%}
[highestScoringTrial, indxHighestScoringTrial]=min(allFitnessScoresOneDummy2Live);
highestAgent1OneDummy2Live = agent1LocationsOneDummy2Live(indxHighestScoringTrial,:);
%recordedAgent = agent2Locations(indxHighestScoringTrial,:);
highestAgent3OneDummy2Live = agent3LocationsOneDummy2Live(indxHighestScoringTrial,:);

subplot(1,2,2)
plot(time,highestAgent1OneDummy2Live)
hold on
plot(time,recordedAgent)
hold on
plot(time,highestAgent3OneDummy2Live,'k')
hold off
legend('Agent One', 'Dummy Agent','Agent Three');
xlabel('Time');
ylabel('Location');
function W = formMatrix(vector,noNodes)
    W= reshape(vector,noNodes,noNodes);
end
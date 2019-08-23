clear;
clf;
%set GA values
NoIndividuals = 40; 
NoWeights = 9;
NoGenerations = 5000;  % might have to make 5000*40
copyRate = 0.5;
NoTrials = 15;
demeSize = 3;
%set CTRNN values
dt = 0.05;
T=25;
time = 0:dt:T;
N = 3;%number of nodes
maxWeight = 8;
minWeight = -8;
maxSensorGain = 100;
minSensorGain = 1;
maxOutputGain = 50;
minOutputGain = 1;
weightsRange = range([minWeight, maxWeight]);
sensorGainRange = range([minSensorGain,maxSensorGain]);
outputGainRange = range([minOutputGain, maxOutputGain]);

%create population
sensorGains = minSensorGain + (maxSensorGain - minSensorGain)*rand(NoIndividuals,1); % sensor gain in range of 1-100
outputGains = minOutputGain + (maxOutputGain - minOutputGain)*rand(NoIndividuals,2);
weights = -8 + (8+8)*rand(NoIndividuals,NoWeights); %9 weights
population = [weights sensorGains outputGains];%population combination of weights and sensor gain
NoGenes = length(population(1,:));
mutationRate = 1/NoGenes;

%initial calculation of fitness
for a = 1:NoIndividuals
    individualW = formMatrix(population(a,1:9),N);
    individualSensorGain = population(a,10);
    individualOutputGain = population(a,11);
    individualOutputGain2 = population(a,12);
    fitness(a) = evaluateFitness(individualW,individualSensorGain,individualOutputGain,individualOutputGain2,N,NoTrials,dt,time);
end

%create random matrices outside loop to save time
randPick1 = rand(1,NoGenerations);
randPick2 = rand(1,NoGenerations);
decideToCopy = rand(1,NoGenerations);


for i=1:NoGenerations
  
    %select two from population at random
    pick1 = floor(randPick1(:,i)*NoIndividuals+1);
    ind1 = population(pick1,:);
    ind1Fitness = fitness(pick1); %get fitness
    
    %second individual chosen from area local to the first individual
    pick2 = (pick1+1+floor(demeSize*randPick2(:,i)));
    if(pick2 <= NoIndividuals)
        ind2 = population(pick2,:);
        ind2Fitness = fitness(pick2);
    %loop around makes sure individual 40 is next to individual 1
    else
        loopAroundPick = (pick2 - NoIndividuals);
        ind2 = population(loopAroundPick,:);
        ind2Fitness = fitness(loopAroundPick);
    end
    
    
    %form matrices from the strings
    ind1Weights = formMatrix(ind1(:,1:9),N);
    ind2Weights = formMatrix(ind2(:,1:9),N);
    
    %assign winner/loser status
    if(ind1Fitness > ind2Fitness)
        winner = ind1;
        loser = ind2; 
    else
        winner=ind2;
        loser=ind1; 
    end
    
    %recombine & mutate loser
    for j=1:NoGenes
       %recombine
       if(decideToCopy(i)<copyRate)
          loser(j) = winner(j);
       end
    end
    for w=1:NoWeights
       %mutate wieghts
       if(rand<mutationRate)
          loser(:,w) =+ weightsRange/100*randn;
       end
       if(loser(:,w) > maxWeight)
           loser(:,w) = loser(:,w) - weightsRange;
       elseif(loser(w) < minWeight)
           loser(:,w) = loser(:,w) + weightsRange;
       end
    end
    if(rand<mutationRate)%mutate sensor and output gains
        loser(:,10) =+ sensorGainRange/100*randn;
        if(loser(:,10) > maxSensorGain)
            loser(:,10) = loser(:,10) - sensorGainRange;
        elseif(loser(10) < minSensorGain)
            loser(:,10) = loser(:,10) + sensorGainRange;
        end
    end
    if(rand<mutationRate)
        loser(:,11)=+ outputGainRange/100*randn;
        if(loser(:,11) > maxOutputGain)
            loser(:,11) = loser(:,11)-outputGainRange;
        elseif(loser(:,11) < minOutputGain)
            loser(:,11) = loser(:,11) + outputGainRange;
        end
    end
    if(rand<mutationRate)
        loser(:,12)=+ outputGainRange/100*randn;
        if(loser(:,12) > maxOutputGain)
            loser(:,12) = loser(:,12)-outputGainRange;
        elseif(loser(:,12) < minOutputGain)
            loser(:,12) = loser(:,12) + outputGainRange;
        end
    end
    
    
    %put loser back in population
    if(pick2 <= NoIndividuals)
        if(winner ==ind1)
            population(pick2,:) = loser;
            fitness(pick2) =  evaluateFitness(ind2Weights,ind2(:,10),ind2(:,11),ind2(:,12),N,NoTrials,dt,time);
            fitness(pick1) = evaluateFitness(ind1Weights,ind1(:,10),ind1(:,11),ind1(:,12),N,NoTrials,dt,time);
        elseif(winner ==ind2)
            population(pick1,:) = loser;
            fitness(pick1) =  evaluateFitness(ind1Weights,ind1(:,10),ind1(:,11),ind1(:,12),N,NoTrials,dt,time);
            fitness(pick2) =  evaluateFitness(ind2Weights,ind2(:,10),ind2(:,11),ind2(:,12),N,NoTrials,dt,time);
        end
    elseif(pick2>NoIndividuals)
        if(winner ==ind1)
            population(loopAroundPick,:) = loser;
            fitness(pick1) =  evaluateFitness(ind1Weights,ind1(:,10),ind1(:,11),ind1(:,12),N,NoTrials,dt,time);
            fitness(loopAroundPick) =  evaluateFitness(ind2Weights,ind2(:,10),ind2(:,11),ind2(:,12),N,NoTrials,dt,time);
        elseif(winner ==ind2)
            population(pick1,:) = loser;
            fitness(pick1) =  evaluateFitness(ind1Weights,ind1(:,10),ind1(:,11),ind1(:,12),N,NoTrials,dt,time);
            fitness(loopAroundPick) =  evaluateFitness(ind2Weights,ind2(:,10),ind2(:,11),ind2(:,12),N,NoTrials,dt,time);
        end
    end
    
  
 [bestFitness(i), bestIndividualIndex] =max(fitness);
 bestFitness(i) = bestFitness(i);
  
end
bestIndividual = population(bestIndividualIndex,:);
bestIndividualWeights = formMatrix(bestIndividual(:,1:9),N);
bestIndividualSensor = bestIndividual(:,10);
bestIndividualOutput1 = bestIndividual(:,11);
bestIndividualOutput2 = bestIndividual(:,12);
finalFitness = bestFitness(end);
subplot(1,4,1)
plot(bestFitness)

function [finalFitness] = evaluateFitness(agentWeights, sensorGain, outputGain,outputGain2,N,NoTrials,dt,time)


    fitnessScores = [];
    noiseSD = sqrt(0.5);
    noiseMean = 0;
    noiseMatrix = noiseSD.*randn(NoTrials,length(time)) + noiseMean;
    noiseMatrix2 = noiseSD.*randn(NoTrials,length(time)) + noiseMean;
    noiseMatrix3 = noiseSD.*randn(NoTrials,length(time)) + noiseMean;
    noiseMatrix4 = noiseSD.*randn(NoTrials,length(time)) + noiseMean;
    noiseMatrix5 = noiseSD.*randn(NoTrials,length(time)) + noiseMean;
    noiseMatrix6 = noiseSD.*randn(NoTrials,length(time)) + noiseMean;
    initialConditionsMatrix = randn(N,NoTrials);
    initialConditionsMatrix2 = randn(N,NoTrials);
    initialConditionsMatrix3 = randn(N,NoTrials);
    startPoints = linspace(-25,25,NoTrials);
    %startPoints2 = linspace(0,0,NoTrials);
    %agent input
    I1=0;
    agentOneI = zeros(N,1);
    agentOneI(:,:) = I1; 
    agentTwoI = zeros(N,1);
    agentThreeI(:,:) = zeros(N,1);
    
    for t = 1:NoTrials
        noise = noiseMatrix(t,:);
        noise2 = noiseMatrix2(t,:);
        noise3 = noiseMatrix3(t,:);
        noise4 = noiseMatrix4(t,:);
        noise5 = noiseMatrix5(t,:);
        noise6 = noiseMatrix6(t,:);


        %set agents up
        %starting conditions - agent one
        agentOne = zeros(N,length(time));
        %agent initial conditions
        agentOne(:,1) = initialConditionsMatrix(:,t);
        %agent2
        agentTwo = zeros(N,length(time));
        agentTwo(:,1) = initialConditionsMatrix2(:,t); 
        %agent Three
        agentThree = zeros(N,length(time));
        agentThree(:,1) = initialConditionsMatrix3(:,t);


        %set start locations
        agent1StartPoint = startPoints(:,t); 
        agentOneLocation = zeros(1,length(time));
        agentOneLocation(:,1) = agent1StartPoint;
        agentTwoLocation = zeros(1,length(time));
        agentTwoLocation(:,1) = 0;%starts at 0
        agentThreeLocation = zeros(1,length(time));
       %agent3StartPoint = 
        agentThreeLocation(:,1) = agent1StartPoint; %agent3 always starts at agent 1 position
        
        crossLocations = [];
        
        [~,~,crossLocations]=runSimulation(agentOne,agentTwo,agentThree,agentOneLocation,agentTwoLocation,agentThreeLocation,agentWeights,agentOneI,agentTwoI,agentThreeI,sensorGain,outputGain,outputGain2,crossLocations,noise,noise2,noise3,noise4,noise5, noise6,time,dt);
        
        if(isempty(crossLocations))
            crossLocation = 0;
        else
            crossLocation = crossLocations(end);  
        end
        
        fitness= (abs(crossLocation));
        fitnessScores = [fitnessScores fitness];
    end
    finalFitness = median(fitnessScores);
    
    
end

function W = formMatrix(vector,noNodes)
    W= reshape(vector,noNodes,noNodes);
end


function [agentOneLocation,agentTwoLocation,crossLocations]=runSimulation(agentOne,agentTwo,agentThree,agentOneLocation,agentTwoLocation,agentThreeLocation,agentWeights,agentOneI,agentTwoI,agentThreeI,sensorGain,outputGain,outputGain2,crossLocations,noise,noise2,noise3,noise4,noise5,noise6,time,dt)
    bla = linspace(-200,0,1000);
    crossLocations = [];
    for i= 2:length(time)
            
            if(rand<0.3)
                agentOneI(:,:)=0;
                agentTwoI(:,:)=0;
                agentThreeI(:,:)=0;
            end
            %integrated equation of CTRNN  agent two
            agentTwo(:,i) =agentTwo(:,i-1) +dt*(-agentTwo(:,i-1)+tanh(agentWeights*agentTwo(:,i-1)+ agentTwoI(:,:)));

            %integrated equation of CTRNN agent one
            agentOne(:,i) = agentOne(:,i-1) +dt*(-agentOne(:,i-1)+tanh(agentWeights*agentOne(:,i-1)+ agentOneI(:,:)));
            
            %integrated equation of CTRNN agent three
            agentThree(:,i) = agentThree(:,i-1) +dt*(-agentThree(:,i-1)+tanh(agentWeights*agentThree(:,i-1)+ agentThreeI(:,:)));
            
            %agent Velocity
            agentTwoVelocityLeft = (agentTwo(2,i) + noise(:,i))*outputGain;
            agentTwoVelocityRight = (agentTwo(3,i) + noise2(:,i))*outputGain;
            agentTwoVelocity = (agentTwoVelocityLeft-agentTwoVelocityRight);
            agentOneVelocityLeft =(agentOne(2,i)+noise3(:,i))*outputGain;
            agentOneVelocityRight = (agentOne(3,i)+noise4(:,i))*outputGain2;
            agentOneVelocity = agentOneVelocityLeft-agentOneVelocityRight;
            agentThreeVelocityLeft = (agentThree(2,i) + noise5(:,i))*outputGain;
            agentThreeVelocityRight = (agentThree(3,i) + noise6(:,i))*outputGain;
            agentThreeVelocity = (agentThreeVelocityLeft-agentThreeVelocityRight);

            %agent location
            agentTwoLocation(:,i) = agentTwoLocation(:,i-1) +(agentTwoVelocity);
            agentOneLocation(:,i) = agentOneLocation(:,i-1) - (agentOneVelocity);%negative as facing other direction
            agentThreeLocation(:,i) = agentThreeLocation(:,i-1)+(agentThreeVelocity);

            %input is distance to other agent mapped between 1 and 0 only
            %when agents are within 0 - 200 units of space to eachother
            %on-off sensing essentially
            distanceBetweenAgentOneTwo = -(abs(agentTwoLocation(:,i) - agentOneLocation(:,i)));
            distanceBetweenAgentOneThree = -(abs(agentThreeLocation(:,i) - agentOneLocation(:,i)));
            distanceBetweenAgentTwoThree = -(abs(agentThreeLocation(:,i) - agentTwoLocation(:,i)));
            if(distanceBetweenAgentOneTwo > -200 && distanceBetweenAgentOneTwo< 0)
                bla =[bla distanceBetweenAgentOneTwo];
                norm_data = (bla - min(bla)) / ( max(bla) - min(bla) );
                agentOneI(1,:) = norm_data(end)*sensorGain;
                agentTwoI(1,:)= norm_data(end)*sensorGain;
            else
                agentOneI(1,:) = 0;
                agentTwoI(1,:)=0;
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
                agentTwoI(2,:) = norm_data(end)*sensorGain;
                agentThreeI(2,:)= norm_data(end)*sensorGain;
            else
                agentTwoI(2,:) = 0;
                agentThreeI(2,:)=0;
            end
            
            agentLocations = [agentOneLocation(:,i),agentTwoLocation(:,i),agentThreeLocation(:,i)];
            if(range(agentLocations)<40)
                crossLocations = [crossLocations agentOneLocation(:,i)];
            end

    end
    
end

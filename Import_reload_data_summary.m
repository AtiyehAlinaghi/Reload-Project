% Script file to go through all the folders on the reload data set
% Read the JSON files of Initial & Daily information for each participant
% Save the data in an Excel spread sheet
% It is designed based on the format and specific field names of Reload
% data sets
%##########################################################################
% Last update 06/03/2025 11:30 a.m.
% Atiyeh Alinaghi, Reload Project
% University of Southampton
%##########################################################################

% Define the parent folder path
% This the folder provided by Cambridge colleagues 
parentFolderPath = 'reload_03_03_2025\'; % Change this to your parent folder path

% Initialize a cell array to store data
allData = {};
% Demographic information titles
% There are 2 sets as the data was saved in both lower & upper case.
demog_Uppercase = ["Cohort", "GP_surgery_name","datetime","User_sex", "User_gender", "User_age","User_weight", "User_height",...
    "User_ethnicity","User_education", "Medical_conditions_A", "Medical_conditions_B", "Medical_conditions_C",...
    "Hospital_admissions", "Smoking", "Use_of_reliever_inhalers_last_month", "Number_of_reliever_inhalers", ...
    "Use_of_preventer_inhalers_last_month", "Number_of_preventer_inhalers", ...
    "Number_of_oral_medications_taken"];

demog_lowercase = ["cohort", "gp_name","datetime", "user_sex", "user_gender", "user_age", "user_weight", "user_height", "user_ethnicity",...
    "user_education","medical_conditions_A","medical_conditions_B","medical_conditions_C",...
    "hospital_history", "smoking","use_of_reliever_inhalers_last_month", "number_of_reliever_inhalers", ...
    "use_of_preventer_inhalers_last_month", "number_of_preventer_inhalers",...
    "number_oral_medications_taken"];

% This is how the daily survey is recorded in the JSON files
daily_Survey = ["Date","currently_with_rti", "has_recorded",...
    "number_of_days_since_onset", "symptom_severity_A",...
    "symptom_severity_B", "consultation__0x1A_", "consultation_B",...
    "temperature", "pulse", "oxygen_saturation",...
    "diagnosis_question", "diagnosis", "rti_medication"];
% I put consultation__0xCE91_  for consultation_A due to ASCI code
% conversion consultation__0x1A_


% The daily report titles as appeared on the report excel spread sheet
daily_report = ["Date",...
    "Do you currently have a respiratory infection (cough or cold)?",...
    "Have you already recorded data for this respiratory tract infection?",...
    "How many days ago did your symptoms of a respiratory tract infection start?",...
    "Please describe each of the following symptoms with a score : Cough",...
    "Please describe each of the following symptoms with a score : Phlegm",...
    "Please describe each of the following symptoms with a score : Shortness of breath",...
    "Please describe each of the following symptoms with a score : Wheeze",...
    "Please describe each of the following symptoms with a score : Blocked/ runny nose",...
    "Please describe each of the following symptoms with a score : Fever",...
    "Please describe each of the following symptoms with a score: Chest pain",...
    "Please describe each of the following symptoms with a score: Muscle ache",...
    "Please describe each of the following symptoms with a score: Headache",...
    "Please describe each of the following symptoms with a score: Disturbed sleep",...
    "Please describe each of the following symptoms with a score: Feeling generally unwell",...
    "Please describe each of the following symptoms with a score: interference with normal activites/ work",...
    "Since you last used the app have you been to hospital because of your respiratory tract infection?",...
    "Since you last used the app have you consulted with a medical professional about your respiratory tract infection?",...
    "Please provide your temperature,  if you know it.",...
    "Please provide your pulse,  if you know it.",...
    "Please provide your oxygen saturation if you know it.",...
    "Have you been given a diagnosis for your current respiratory tract infection by a medical professional?",...
    "What was the diagnosis?",...
    "Are you taking any medication for your respiratory tract infection?"];




d = 1;
allData{end+1, d} = 'Participant ID';
for d = 1:length(demog_Uppercase)
    allData{end, d+1} = demog_Uppercase {d};
end
L_demog = length(demog_Uppercase);
Num_days = 20; % Maximum number of days that the RTI will last
Num_symptoms = length(daily_report);
for day = 1:Num_days
    for symp = 1:Num_symptoms
        allData{end, L_demog+((Num_symptoms)*(day-1))+1+symp} = "Day "+ num2str(day) + ": " + daily_report(symp);
    end
end



% List all the folders in the parent folder
cd(parentFolderPath);
participants = dir('reload\');
num_folder = length(participants); % Number of participants
% Find indices of '.' and '..'
dotIndices = ismember({participants.name}, {'.', '..'});

% Filter out '.' and '..'
participants(dotIndices) = [];

% Loop through each participants folder
for p = 1:numel(participants)
    user_id = participants(p).name;
    % Figure out whether Android or iOS.
    if length(user_id) == 10
        device = "Android";
    elseif length(user_id) == 12
        device = "iOS";
    end
    disp(participants(p).name);
    participantFolder = participants(p).name;
    participantFolderPath = fullfile('reload\', participantFolder);

    % List all folders in the participants folder
    EachCaseFolders = dir(participantFolderPath);
    dotIndices = ismember({EachCaseFolders.name}, {'.', '..'});
    % Filter out '.' and '..'
    EachCaseFolders(dotIndices) = [];

    allData{end+1, 1} = participantFolder; % The participant's ID

    has_signup = false; justSignUp = false;
    % Number of folders in each participant's folder
    NumOfDays = numel(EachCaseFolders);


    % Loop through each folder, i.e. each day, in each participant folder
    for f = 1:numel(EachCaseFolders) % Day f (Day counter)
        folderName = EachCaseFolders(f).name; % Including the date
        folderPath = fullfile(participantFolderPath, folderName);


        % List all JSON files in the folder
        jsonFiles = dir(fullfile(folderPath, '*.json'));


        % Loop through each JSON file in the folder
        for j = 1:numel(jsonFiles)
            jsonFileName = jsonFiles(j).name;
            jsonFilePath = fullfile(folderPath, jsonFileName);

            % Some folders only have signUp json file without daily report
            if numel(jsonFiles) < 2 && or(contains(jsonFileName,'signUp'),contains(jsonFileName, 'initial'))
                justSignUp = true;
            end

            % Store information in cell array


            addpath jsonlab-master\
            % Read JSON file and extract information
            % Assuming you have JSONLab or similar library for JSON file reading
            json_data = loadjson(jsonFilePath); 

            % Check if this file is the initial signup survey.
            if or(contains(jsonFileName,'signUp'),contains(jsonFileName, 'initial'))
                has_signup = true;
            elseif contains(jsonFileName,'daily')
                has_daily = true;
                if justSignUp
                    day = f-1;
                else
                    day = f;
                end
            end


            % Get info from signup survey.
            if has_signup
                if isfield(json_data, "Consent")
                    demog = demog_Uppercase;
                else
                    demog = demog_lowercase;
                end

                if isfield(json_data,"chronic_respiratory_condition")
                    conditions = json_data.("chronic_respiratory_condition");
                    parts = strsplit(conditions, ';');
                    % Medical_conditions_A, B & C
                end

                if isfield(json_data,"use_of_inhalers_last_month" )
                    inhales = json_data.("use_of_inhalers_last_month");
                    inhale = strsplit(inhales, ';');
                    
                end


                for i = 2:length(demog)+1
                    fieldname_init = demog {i-1};
                    if isfield(json_data, fieldname_init)
                        if strcmp(fieldname_init,"datetime")
                            dt = json_data.(fieldname_init);
                            if strcmp(device,"Android")
                                time_stamp = int32(double(dt/1000));
                            elseif strcmp(device,"iOS")
                                time_stamp = int32(double(floor(str2double(dt))));
                            end
                            allData{end, i} = datestr(datetime(time_stamp,'ConvertFrom', 'posixtime'));
                        elseif ((i== length(demog)+1) || strcmp(fieldname_init,"user_education") || ...
                                strcmp(fieldname_init,"User_education")) % && strcmp(device,"Android")
                            % The range of education years or number of
                            % oral medication
                            num_range = json_data.(fieldname_init);
                            % to ensure that the resulting string is treated as literal text rather than 
                            % being interpreted in any other way, such as a date or a number.
                            allData{end, i} = strcat("'",num_range);

                        elseif ((i==8) || (i==9)) % User's weight & height
                            if strcmp(json_data.(fieldname_init),'-40') ||strcmp(json_data.(fieldname_init),'-150')
                                below_val = eval(json_data.(fieldname_init));
                                allData{end, i} = strcat("'below ",num2str(abs(below_val)));
                            else
                                allData{end, i} = json_data.(fieldname_init);
                            end
                        elseif (strcmp(fieldname_init,"number_of_reliever_inhalers") || ...
                                strcmp(fieldname_init,"Number_of_reliever_inhalers")|| ...
                                strcmp(fieldname_init,"number_of_preventer_inhalers")|| ...
                                strcmp(fieldname_init,"Number_of_preventer_inhalers"))
                            num_range_inhaler = json_data.(fieldname_init);
                            % to ensure that the resulting string is treated as literal text rather than 
                            % being interpreted in any other way, such as a date or a number.
                            allData{end, i} = strcat("'",num_range_inhaler);
                            if strcmp(json_data.(fieldname_init),'-1')
                                allData{end, i} = "Less than one";
                            end
                        else
                            allData{end, i} = json_data.(fieldname_init);
                        end
                    elseif isfield(json_data,"chronic_respiratory_condition") && ((i==12) || (i==13) || (i==14))
                        % "Medical_conditions_A", "Medical_conditions_B", "Medical_conditions_C"
                        allData{end, i} = parts{i-11}; 
                        % parts are the conditions separated by ";"                         
                        % It splits the group of medical conditions into
                        % separate condiitons 
                    elseif isfield(json_data,"use_of_inhalers_last_month") && ((i==17) || (i==18))
                        % "Use_of_reliever_inhalers_last_month" or "Number_of_reliever_inhalers"
                        if size(inhale,2) == 4
                            % inhale contains the answers split by ';'
                            if (i==17)
                                allData{end, i} = inhale{2};
                            elseif (i==18)
                                allData{end, i} = inhale{4};
                            end

                        elseif size(inhale,2) == 2
                            inhale_ans = inhale{i-16};
                            divide_ans = strsplit(inhale_ans,',');
                            if strcmp(divide_ans{1}, 'Yes')
                                if strcmp(divide_ans{2}, '-1')
                                    allData{end, i} = "less that once";
                                else
                                    allData{end, i} = divide_ans{2};
                                end
                            else
                                allData{end, i} = strcat("'",divide_ans{1});
                            end
                        end
                    end

                end

                has_signup = false;

            elseif has_daily
                dt = json_data.("datetime");
                if strcmp(device,"Android")
                    time_stamp = int32(double(dt/1000));
                elseif strcmp(device,"iOS")
                    time_stamp = int32(double(floor(str2double(dt))));
                end
                allData{end, L_demog+((Num_symptoms)*(day-1))+1+1} = ...
                    datestr(datetime(time_stamp,'ConvertFrom', 'posixtime'));
                % day = day+1;
                RTI = "currently_with_rti";
                if isfield(json_data, RTI)
                    symps = 2;
                    allData{end, L_demog+((Num_symptoms)*(day-1))+1+symps} =...
                        json_data.("currently_with_rti");
                elseif isfield(json_data, "RTI")
                    allData{end, L_demog+((Num_symptoms)*(day-1))+1+symps} =...
                        json_data.("RTI");
                end

                for survey = 3:length(daily_Survey)
                    % Going through all the daily quesitons' fields
                    fieldname_daily = daily_Survey{survey};
                    if isfield(json_data,fieldname_daily)
                        if strcmp(fieldname_daily,"symptom_severity_A")
                            symptomsA = json_data.(fieldname_daily);
                            split_sympA = split(symptomsA, ';');
                            for s = 1: length(split_sympA)
                                allData{end, L_demog+((Num_symptoms)*(day-1))+1+survey+s-1} = split_sympA{s};
                            end
                        elseif strcmp(fieldname_daily,"symptom_severity_B")
                            symptomsB = json_data.(fieldname_daily);
                            split_sympB = split(symptomsB, ';');
                            for s = 1: length(split_sympB)
                                allData{end, L_demog+((Num_symptoms)*(day-1))+1+2*(survey-1)+s} = split_sympB{s};
                            end
                        end
                        if survey < 5
                            allData{end, L_demog+((Num_symptoms)*(day-1))+1+survey} = json_data.(fieldname_daily);
                        elseif survey > 6
                            % Because we report all the symptoms' scores
                            % separately we need to jump Num = 10 columns
                            % for them
                            allData{end, L_demog+((Num_symptoms)*(day-1))+1+survey+10} = json_data.(fieldname_daily);
                        end
                    end
                end

                has_daily = false;

            end


            % Add additional data fields as needed from the 'data' variable

            % Example: If 'data' contains a field named 'info':
            % allData{end, 4} = data.info;
        end
    end
end

% Write data to Excel file
excelFileName = 'Data_Reload_03032025_daily_update.xlsx'; % Name of the Excel file to be created
xlswrite(excelFileName, allData);
disp(['Data saved to ', excelFileName]);

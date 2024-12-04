function dicomsort(pathname)

% Sort all dicom files in a directory tree into
% separate folders based on study+series UIDs.

if ~exist('pathname','var') || isempty(pathname)
    pathname = uigetdir();
    if isequal(pathname,0); return; end
end
if pathname(end) ~= filesep
    pathname(end+1) = filesep;
end

info = get_files(pathname);
n = numel(info);

outdir = tempname;
if outdir(end) ~= filesep
    outdir(end+1) = filesep;
end

fprintf('%s.m: writing to %s\b ',mfilename,outdir);
fprintf(repmat(' ',floor(log10(n))+4,1));

% counter (use file to handle parfor)
countfile = tempname;
fid = fopen(countfile,'w+');
fwrite(fid,zeros(n,1,'uint8'));
fclose(fid);

for j = 1:n
    
    filename = [pathname info(j).name];
    
    % skip non-DICOM files
    if isdicom(filename)    

        % DICOM header
        hdr = dicominfo(filename);
        
        StudyInstanceUID = hdr.StudyInstanceUID;
        SeriesInstanceUID = hdr.SeriesInstanceUID;
        
        % unique name for study and series (if both empty then skip)
        if ~isempty(StudyInstanceUID) || ~isempty(SeriesInstanceUID)
            
            %if ~isempty(StudyInstanceUID) && ~isempty(SeriesInstanceUID)
            %    mydir = [outdir StudyInstanceUID '.' SeriesInstanceUID];
            %else 
            %    mydir = [outdir StudyInstanceUID SeriesInstanceUID];
            %end
            if ~isfield(hdr,'StudyDescription')
                hdr.StudyDescription = hdr.StudyID;
            end
            if ~isfield(hdr,'SeriesDescription')
                hdr.SeriesDescription = [hdr.SeriesDate hdr.SeriesTime];
            end

            % directory name
            mydir = [outdir hdr.StudyDescription filesep [hdr.SeriesDescription ' series ' num2str(hdr.SeriesNumber)]];
            mydir = strrep(mydir,' ','_');

            % make a hash from the unique identifiers
            %key = [StudyInstanceUID SeriesInstanceUID];
            %key = keyHash(key);            
            %mydir = [mydir '__uid' num2str(key)];

            if ~exist(mydir,'dir')
                [~,~] = mkdir(mydir);
            end

            copyfile(filename,mydir);
        
            % update counter
            fid = fopen(countfile,'r+');
            count = sum(fread(fid));
            fseek(fid,j-1,'bof');
            fwrite(fid,1); fclose(fid);
            fprintf(sprintf('\b\b\b\b\b\b%-5d\n',n-count));
        end
        
    end
    
end
fprintf(sprintf('\b\b\b\b\b\b\n'));

% report non-DICOM files
fid = fopen(countfile,'r');
ok = fread(fid); fclose(fid);
delete(countfile);

not_ok = ~ok;
for j = 1:sum(not_ok)
    fprintf('failed: %s\n',info(not_ok(j)).name);
end

%% get all dicom files in a directory tree
function info = get_files(pathname)

% contents of starting directory
info = dir(pathname);

j = 1;
while j <= numel(info)
    % recurse into subdirectories
    if info(j).isdir
        % skip subdirectories with a leading '.'
        if info(j).name(1) ~= '.'
            temp = dir([pathname info(j).name]);
            % prepend path (except for '.')
            for k = 1:numel(temp)
                if temp(k).name(1) ~= '.'
                    temp(k).name = [info(j).name filesep temp(k).name];
                end
            end
            % append contents of subdirectory
            info = [info;temp];
        end
        % delete directory from list
        info(j) = [];
    else
        % skip past files
        j = j + 1;
    end
end

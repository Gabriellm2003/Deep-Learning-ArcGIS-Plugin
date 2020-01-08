import arcpy
import subprocess
from subprocess import Popen, PIPE
import sys
import os
import tempfile
import re
import shutil



def removeEmptyFolders(path, removeRoot=True):
    'Function to remove empty folders'
    if not os.path.isdir(path):
        return

    if removeRoot and not ('erosion' in path):
        return

    # remove empty subfolders
    files = os.listdir(path)
    if len(files):
        for f in files:
            fullpath = os.path.join(path, f)
            if os.path.isdir(fullpath):
                removeEmptyFolders(fullpath)

    # if folder empty, delete it
    files = os.listdir(path)
    if len(files) == 0 and removeRoot and ('erosion' in path):
        # print("Removing empty folder:", path)
        os.rmdir(path)

def removeOldModels(path):
    'Function to remove empty folders'
    if not os.path.isdir(path):
        return

   
    # remove empty subfolders
    folders = os.listdir(path)  
    if len(folders):
        for f in folders:
            fullpath = os.path.join(path, f)
            if os.path.isdir(fullpath):
                models = [x for x in os.listdir(fullpath) if '.h5' in x]
                models.sort()
                for m in range(len(models) - 1):
                    print(os.path.join(fullpath, models[m]))
                    os.remove(os.path.join(fullpath, models[m]))

class FineTunningSeg(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Improve Model (Erosion)"
        self.description = "Tool for improve (fine tune) the erosion segmentation model."
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        rasterLayers = arcpy.Parameter(
                        displayName="Base images of annotations",
                        name="imgs",
                        datatype="GPRasterLayer",
                        parameterType="Required",
                        direction="Input",
                        multiValue=True
                        )
        rlwShapefile = arcpy.Parameter(
                        displayName="Railway track's shapefile",
                        name="rlwshpfile",
                        datatype="GPFeatureLayer",
                        parameterType="Required",
                        direction="Input"
                        )        
        annotations = arcpy.Parameter(
                        displayName="Erosions' annotations",
                        name="shpfile",
                        datatype="GPFeatureLayer",
                        parameterType="Required",
                        direction="Input"
                        )

        params = [rasterLayers, annotations, rlwShapefile]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        names = []
        sameMachine = True
        debug = [False, False]

        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#         os.environ["CUDA_VISIBLE_DEVICES"]="1"
        
        '''
        SCRIPTS_PATH = os.path.dirname(__file__)
        ROOT_PATH = os.path.abspath("..")
        #messages.addMessage("PATHS")
        #messages.addMessage(SCRIPTS_PATH)
        #messages.addMessage(ROOT_PATH)
        ROOT_PATH = os.path.split(SCRIPTS_PATH)[0]
        #messages.addMessage(ROOT_PATH)
        sys.path.append(SCRIPTS_PATH)
        '''
        ROOT_PATH = os.path.join(os.path.dirname(__file__),"src")
        sys.path.append(ROOT_PATH)

        checkShapeCmd = os.path.join(ROOT_PATH, "utils\\checkShapeFile.py")
        prepCmd = os.path.join(ROOT_PATH, "utils\\createDatasets.py") # Script de preprocessamento dos dados para a MaskRcnn
        netCmd = os.path.join(ROOT_PATH, "mrcnn\\tcu_model\\erosion_detection.py") # Script da rede (MaskRcnn)
        # pythonCmd = os.path.join('C:\\Python352','python.exe')
        path_variables = os.environ['path']
        path_variables = path_variables.split(';')
        candidates = [a for a in path_variables if os.path.isfile(a+"\\python.exe")]
        regex = r".*\(major=(\d), minor=(\d), micro=(\d)"
        foundPython = False
        for i in candidates:
            ret = subprocess.check_output(i + '\\python.exe -c "import sys; print(sys.version_info)"')
            m = re.search(regex, ret)
            if int(m.group(1)) == 3 and int(m.group(2)) == 5 and int(m.group(3)) == 2:
                # print "esse eh o python 3.5.2"
                pythonCmd = os.path.join(i,'python.exe')
                foundPython = True
        if not foundPython:
            messages.addMessage("\n*****\n***You must have Python 3.5.2 installed to run the plugin.***\n*****\n")
            return 
        # pythonCmd = os.path.join(os.path.dirname(sys.executable),'python.exe')

        rlwshapefile = "rlwshp_placeholder"
        outputFolder = "model1" 
        
        modelWeights = os.path.join(ROOT_PATH, "mrcnn\\tcu_model\\model1")

        outputFolder = modelWeights

        # Raliway Shapefile
        d = arcpy.Describe(parameters[2].value)
        rlwShpfilePath = d.catalogPath
        #messages.addMessage(d.catalogPath)

        # Shapefile
        d = arcpy.Describe(parameters[1].value)
        annotationsPath = d.catalogPath
        #messages.addMessage(d.catalogPath)

        # Imagens
        imgfiles = []
        for raster in parameters[0].values:
            d = arcpy.Describe(raster)
            if sameMachine:
                imgfiles.append(d.catalogPath)
            else:
                imgfiles.append(d.catalogPath.replace('\\', '/'))
        '''
        Procedimento eh o mesmo:
        - Para cada img passada no input:
            - cria uma pasta temporaria para armazenar arquivos usados de input na rede
            - preprocessa gerando crops das imgs usando o shapefile(enviando para o server se preciso)
              e salvando na pasta temporaria
            - envia para a cnn para finetunning, que salva o modelo em uma pasta padrao a ser definida
        '''

        '''
        Assumindo tudo em uma mesma maquina
        '''
        if sameMachine:
            #Check if ShapeFile has only lines
            cmd = []
            cmd.append(pythonCmd)
            cmd.append(checkShapeCmd)
            cmd.append(rlwShpfilePath)
            if debug[0]:
                messages.addMessage(" ".join(cmd))
            else:
                p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                output, err = p.communicate(b"")
                rc = p.returncode
                if output != 'MULTILINESTRING':
                    messages.addMessage("\n*****\n***Check the railway shapefile, it seems it contains more than lines.***\n*****\n")
                    return
            # Preprocess
            cmd = []
            inputFolder = tempfile.mkdtemp()
            cmd.append(pythonCmd)
            cmd.append(prepCmd)             # Script de preprocess 
            ## Parametros 
            cmd.append(",".join(imgfiles))  # Arquivo de img(s) usado para tirar os crops            
            cmd.append(rlwShpfilePath)      # Shapefile da ferrovia, usado como guia para criar os crops
            cmd.append(annotationsPath)     # Shapefile path
            cmd.append("448")               # Tamanho dos crops a ser gerados. 448x448
            cmd.append("train")             # Fine tunning modelo
            cmd.append(inputFolder)         # Pasta para o output dos crops gerados
            cmd.append("segmentation")      # Pasta para o output dos crops gerados
            cmd.append("false")             # Pasta para o output dos crops gerados
             

            if debug[0]:
                messages.addMessage(" ".join(cmd))
            else:
                preProcess = subprocess.call(cmd)
                #messages.addMessage(" ".join(cmd))
                #preProcess = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                #preProcess.wait()


                #messages.addMessage("PREP OUT")
                #for line in preProcess.stdout:
                   #messages.addMessage(line)
                #messages.addMessage("PREP ERROR")
                #for line in preProcess.stderr:
                   #messages.addMessage(line)

            # Fine Tunning
            cmd = []
            cmd.append(pythonCmd)
            cmd.append(netCmd)       # Script da cnn (abaixo usando como padrao a MaskRCNN)
            cmd.append("fnt")        # Procedimento que a rede ira executar
            cmd.append(os.path.join(inputFolder, 'Train'))      # Pasta com as imgs de treino
            cmd.append(os.path.join(inputFolder, 'Validation')) # Pasta com as imgs de validacao
            cmd.append("_")          # Pasta com as imgs para teste
            cmd.append(outputFolder) # Pasta padrao que recebe as saidas(shapefiles e predicoes em img)
            cmd.append("0.001")      # Learning rate
            cmd.append("[20]")       # Epoch numbers
            cmd.append(modelWeights) # Pesos do modelo treinado
            cmd.append("False")      # Criar um mapa de predicao completo(nao usado)
            cmd.append("_")      # Criar um mapa de predicao completo(nao usado)


            if debug[1]:
                messages.addMessage(" ".join(cmd))
            else:
                infProcess = subprocess.call(cmd)
                #infProcess = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                #infProcess.wait()
                

                #messages.addMessage(" ".join(cmd))
                #messages.addMessage("MRCNN OUT")
                #for line in infProcess.stdout:
                   #messages.addMessage(line)
                #messages.addMessage("MRCNN ERROR")
                #for line in infProcess.stderr:
                   #messages.addMessage(line)


            removeEmptyFolders(outputFolder, False)
            removeOldModels(outputFolder)
            shutil.rmtree(inputFolder)

        return 

class AnalyzeImgsSeg(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Analyze Images (Erosion)"
        self.description = "Tool for analyze images and find erosion ."
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        rasterLayers = arcpy.Parameter(
                        displayName="Images for analyze",
                        name="imgs",
                        datatype="GPRasterLayer",
                        parameterType="Required",
                        direction="Input",
                        multiValue=True
                        )
        rlwShapefile = arcpy.Parameter(
                        displayName="Railway track's shapefile",
                        name="rlwshpfile",
                        datatype="GPFeatureLayer",
                        parameterType="Required",
                        direction="Input"
                        ) 
        outName = arcpy.Parameter(
                        displayName="Output layer's name",
                        name="outname",
                        datatype="GPString",
                        parameterType="Required",
                        direction="Input"
                        )
        outFolder = arcpy.Parameter(
                        displayName="Output folder's path",
                        name="outfolder",
                        datatype="DEFolder",
                        parameterType="Required",
                        direction="Input"
                        )

        outFolder.value = os.environ.get('USERPROFILE')

        params = [rasterLayers, rlwShapefile, outName, outFolder]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        if parameters[2].altered:
            if not all(ord(c) < 128 for c in parameters[2].valueAsText):
                parameters[2].setErrorMessage("Names with accent are not allowed.")
        if parameters[3].altered:
            if not all(ord(c) < 128 for c in parameters[3].valueAsText):
                parameters[3].setErrorMessage("Paths with accent are not allowed.")
        return


    

    def execute(self, parameters, messages):
        """The source code of the tool."""
        sameMachine = True
        debug = [False, False]
        names = []
        '''
        SCRIPTS_PATH = os.path.dirname(__file__)
        ROOT_PATH = os.path.abspath("..")
        #messages.addMessage("PATHS")
        #messages.addMessage(SCRIPTS_PATH)
        #messages.addMessage(ROOT_PATH)
        ROOT_PATH = os.path.split(SCRIPTS_PATH)[0]
        #messages.addMessage(ROOT_PATH)

        sys.path.append(SCRIPTS_PATH)
        '''
        ROOT_PATH = os.path.join(os.path.dirname(__file__),"src")
        sys.path.append(ROOT_PATH)
        
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#         os.environ["CUDA_VISIBLE_DEVICES"]="1"

        #messages.addMessage(os.environ)

        imgsplitCmd = os.path.join(ROOT_PATH, "utils\\imgSpliter.py")
        checkShapeCmd = os.path.join(ROOT_PATH, "utils\\checkShapeFile.py")
        prepCmd = os.path.join(ROOT_PATH, "utils\\createDatasets.py") # Script de preprocessamento dos dados para a MaskRcnn
        netCmd = os.path.join(ROOT_PATH, "mrcnn\\tcu_model\\erosion_detection.py") # Script da rede (MaskRcnn)
        pythonCmd = '' 
        path_variables = os.environ['path']
        path_variables = path_variables.split(';')
        candidates = [a for a in path_variables if os.path.isfile(a+"\\python.exe")]
        regex = r".*\(major=(\d), minor=(\d), micro=(\d)"
        foundPython = False
        for i in candidates:
            ret = subprocess.check_output(i + '\\python.exe -c "import sys; print(sys.version_info)"')
            m = re.search(regex, ret)
            if int(m.group(1)) == 3 and int(m.group(2)) == 5 and int(m.group(3)) == 2:
                # print "esse eh o python 3.5.2"
                pythonCmd = os.path.join(i,'python.exe')
                foundPython = True
        if not foundPython:
            messages.addMessage("\n*****\n***You must have Python 3.5.2 installed to run the plugin.***\n*****\n")
            return
        # pythonCmd = os.path.join(os.path.dirname(sys.executable),'python.exe')

        rlwshapefile = "rlwshp_placeholder"
        
        #modelWeights = os.path.join(ROOT_PATH, "mrcnn\\tcu_model\\model1\\erosion20180715T2313\\mask_rcnn_erosion_0160.h5")
        modelWeights = os.path.join(ROOT_PATH, "mrcnn\\tcu_model\\model1")
    
        #outputFolder = os.path.abspath("tcuAnalyzes") 
        outputFolder = parameters[3].valueAsText 
        outputShape = parameters[2].valueAsText + '.shp'
        

        # Crete output folder if necessary
        if not os.path.exists(outputFolder):
            os.mkdir(outputFolder)
            os.chmod(outputFolder, 7777)
        
        # Railway Shapefile
        d = arcpy.Describe(parameters[1].value)
        rlwShpfilePath = d.catalogPath
        #messages.addMessage(d.catalogPath)

        # Imgs
        imgfiles = []
        for raster in parameters[0].values:
            d = arcpy.Describe(raster)
            imgfiles.append(d.catalogPath)


        '''
        Procedimento eh o mesmo:
        - Para cada img passada no input:
            - cria uma pasta temporaria para armazenar arquivos usados de input na rede
            - preprocessa gerando crops da img e salvando na pasta temporaria
            - envia para a cnn para inferencia, que salva os outputs em uma pasta padrao a ser definida
            - TEMP a rede retorna a img predita completa que eh mostrada no arcmap
            - TODO a rede retorna um shapefile que eh carregado no arcmap
        '''

        '''
        Assumindo tudo em uma mesma maquina
        '''
        if sameMachine:
            #Check if ShapeFile has only lines
            cmd = []
            cmd.append(pythonCmd)
            cmd.append(checkShapeCmd)
            cmd.append(rlwShpfilePath)
            if debug[0]:
                messages.addMessage(" ".join(cmd))
            else:
                p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                output, err = p.communicate(b"")
                rc = p.returncode
                if output != 'MULTILINESTRING':
                    messages.addMessage("\n*****\n***Check the railway shapefile, it seems it contains more than lines.***\n*****\n")
                    return

            cmd = []
            cmd.append(pythonCmd)
            cmd.append(imgsplitCmd)
            cmd.append(','.join(imgfiles))

            if debug[0]:
                messages.addMessage(" ".join(cmd))
            else:
                p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                output, err = p.communicate(b"")
                rc = p.returncode
                messages.addMessage("output")
                messages.addMessage(output)
                imgs = eval(output)  

            outshapes = []
            #messages.addMessage('images '+str(imgs))
            for num, file in enumerate(imgs):
                # Preprocess
                cmd = []
                inputFolder = tempfile.mkdtemp()
                cmd.append(pythonCmd)
                cmd.append(prepCmd)      # Script de preprocess 
                ## Parametros 
#                 cmd.append("test")         # Analisar img
#                 cmd.append("_")            # Shapefile path nao eh usado no teste
#                 cmd.append(rlwShpfilePath) # Shapefile da ferrovia, usado como guia para criar os crops
#                 cmd.append("448")          # Tamanho dos crops a ser gerados. 448x448
#                 cmd.append(inputFolder)    # Pasta para o output dos crops gerados
                cmd.append(file)                # Arquivo de img(s) usado para tirar os crops 
#                 cmd.append(",".join(imgfiles))  # Arquivo de img(s) usado para tirar os crops            
                cmd.append(rlwShpfilePath)      # Shapefile da ferrovia, usado como guia para criar os crops
                cmd.append(rlwShpfilePath)                 # Shapefile path nao eh usado no teste
                cmd.append("448")               # Tamanho dos crops a ser gerados. 448x448
                cmd.append("test")              # Fine tunning modelo
                cmd.append(inputFolder)         # Pasta para o output dos crops gerados
                cmd.append("segmentation")      # Pasta para o output dos crops gerados
                cmd.append("false")             # Pasta para o output dos crops gerados

                #messages.addMessage(os.path.dirname(__file__))
                #messages.addMessage(" ".join(cmd))
                if debug[0]:
                    messages.addMessage(" ".join(cmd))
                else:
                    # preProcess = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    # preProcess.wait()
                    preProcess = subprocess.call(cmd)
                    # #messages.addMessage("PREP OUT")
                    # for line in preProcess.stdout:
                    #    #the real code does filtering here
                    #    #messages.addMessage(line)
                    # #messages.addMessage("PREP ERROR")
                    # for line in preProcess.stderr:
                    #    #the real code does filtering here
                    #    #messages.addMessage(line)
                #inputfolders.append(inputFolder)

                # Inference

                cmd = []
                cmd.append(pythonCmd)
                cmd.append(netCmd)                  # Script da cnn (abaixo usando como padrao a MaskRCNN)
                cmd.append("testing")               # Procedimento que a rede ira executar
                cmd.append("_")                     # Pasta com as imgs de treino
                cmd.append("_")                     # Pasta com as imgs de validacao
                cmd.append(inputFolder)  # Pasta com as imgs para teste
                cmd.append(outputFolder)            # Pasta padrao que recebe as saidas(shapefiles e predicoes em img)
                cmd.append("0")                     # Learning rate
                cmd.append("[0,0,0]")               # Epoch numbers
                cmd.append(modelWeights)            # Pesos do modelo treinado
                cmd.append("True")                  # Criar um mapa de predicao completo(por padrao o teste ja realiza)
                cmd.append(outputShape.replace('.shp', '') + '_' + str(num))             # Nome shapefile de output

                outshapes.append(os.path.join(outputFolder, outputShape.replace('.shp', '') + '_' + str(num) ))

                if debug[1]:
                    messages.addMessage(" ".join(cmd))
                else:
                    #messages.addMessage(" ".join(cmd))
                    #infProcess = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    #infProcess.wait()
                    infProcess = subprocess.call(cmd)
                    ##messages.addMessage("MRCNN OUT")
                    #for line in infProcess.stdout:
                        #the real code does filtering here
                    #    #messages.addMessage(line)
                    ##messages.addMessage("MRCNN ERROR")
                    #for line in infProcess.stderr:
                        #the real code does filtering here
                    #    #messages.addMessage(line)

                # shutil.rmtree(inputFolder)

            cmd = []
            cmd.append(pythonCmd)
            cmd.append(os.path.join(ROOT_PATH,'mrcnn\\tcu_model','ogrmerge.py'))
            cmd.append('-o')
            cmd.append(os.path.join(outputFolder,outputShape))
            #messages.addMessage("OUTSHAPES = " + str(outshapes))
            for shp in outshapes:
                cmd.append(shp + '.shp')
            cmd.append('-single')
            #messages.addMessage(" ".join(cmd))
            merge = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            merge.wait()

            # Apagar os shapefiles usados uniao
            for shp in outshapes:

                # driver.DeleteDataSource(shp)
                nameShp = shp
                os.remove(nameShp+'.shp')
                os.remove(nameShp+'.dbf')
                os.remove(nameShp+'.prj')
                os.remove(nameShp+'.shx')
            for num, file in enumerate(imgs):
                if '_splitpart' in file:
                    os.remove(file)
                    nameFile = file.replace('.img', '')
                    try:
                        os.remove(nameFile+'.ige')
                    except:
                        pass

            outputShapePath = os.path.join(os.path.abspath(outputFolder),outputShape)
            if os.path.exists(outputShapePath):
                arcpy.MakeFeatureLayer_management(outputShapePath, parameters[2].valueAsText)
                mxd = arcpy.mapping.MapDocument('current')
                df = arcpy.mapping.ListDataFrames(mxd)[0]

                layer = arcpy.mapping.Layer(parameters[2].valueAsText)
                arcpy.mapping.AddLayer(df, layer, "AUTO_ARRANGE")
                arcpy.RefreshActiveView()
            else:
                messages.addMessage("No erosion found.")

            
            removeEmptyFolders(outputFolder, False)
            # shutil.rmtree(inputFolder)


        return
# -*- coding: utf-8 -*-
import arcpy
import subprocess
from subprocess import Popen, PIPE
import sys
import os
import tempfile
import shutil
import re

# cmd = 'mode 50, 5'
# os.system(cmd)



class FineTunningDet(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Improve Model (Bridge Detection)"
        self.description = "Tool for improve (fine tune) the railway bridge detection model."
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        rasterLayers = arcpy.Parameter(
                        displayName="Base Images for annotations",
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
                        displayName="Railway Bridges' annotations",
                        name="shpfile",
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

        params = [rasterLayers, annotations, rlwShapefile, outName, outFolder]
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
        if parameters[3].altered:
            if not all(ord(c) < 128 for c in parameters[3].valueAsText):
                parameters[3].setErrorMessage("Paths with accent are not allowed.")
        if parameters[4].altered:
            if not all(ord(c) < 128 for c in parameters[4].valueAsText):
                parameters[4].setErrorMessage("Names with accent are not allowed.")
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        names = []
        sameMachine = True
        debug = [False, False]

        #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        #os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
        ROOT_PATH = os.path.join(os.path.dirname(__file__), "src")
        sys.path.append(ROOT_PATH)

        checkShapeCmd = os.path.join(ROOT_PATH, "utils\\checkShapeFile.py")
        prepCmd = os.path.join(ROOT_PATH, "utils\\createDatasets.py") # Script de preprocessamento dos dados para a MaskRcnn
        netCmd = os.path.join(ROOT_PATH, "faster\\tf-faster-rcnn\\tools\\trainval_net.py") # Script da rede (FasterRCNN)
        posCmd = os.path.join(ROOT_PATH, "faster\\handle_shapefile.py") # Script para gerar o shapefile final

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
        

        # rlwshapefile = "rlwshp_placeholder"
        # outputFolder = "tcuAnalyzes"
        # if not os.path.exists(outputFolder):
        #     os.mkdir(outputFolder)
        #     os.chmod(outputFolder, 7777)

        # Nao precisa definiir o modelo, o mask trata internamente o melhor modelo pra carregar
        # modelWeights = os.path.join(ROOT_PATH, "faster\\tf-faster-rcnn\\output\\vgg16\\tcu_trainval\\default\\")

        # Imagens
        imgfiles = []
        for raster in parameters[0].values:
            d = arcpy.Describe(raster)
            if sameMachine:
                imgfiles.append(d.catalogPath)
            else:
                imgfiles.append(d.catalogPath.replace('\\', '/'))

        # Shapefile
        d = arcpy.Describe(parameters[1].value)
        annotationsPath = d.catalogPath
        #messages.addMessage(d.catalogPath)

        # Railway Shapefile
        d = arcpy.Describe(parameters[2].value)
        rlwShpfilePath = d.catalogPath
        #messages.addMessage(d.catalogPath)

        outputShape = parameters[3].valueAsText + '.shp'
        outputFolder = parameters[4].valueAsText

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
                # if output != 'MULTILINESTRING':
                #     messages.addMessage("\n*****\n***Check the railway shapefile, it seems it contains more than lines.***\n*****\n")
                #     return
            # Preprocess
            cmd = []
            inputFolder = tempfile.mkdtemp()
            cmd.append(pythonCmd)
            cmd.append(prepCmd)             # Script de preprocess
            ## Parametros
            cmd.append(",".join(imgfiles))  # Lista de imagens
            cmd.append(rlwShpfilePath)      # Shapefile da ferrovia, usado como guia para criar os crops
            cmd.append(annotationsPath)     # Shapefile das marcacoes de ponte
           # cmd.append("_")
            cmd.append("448")               # Tamanho dos crops a ser gerados. 448x448
            cmd.append("train")             # Flag para geracao do dataset (treino ou teste)
            #cmd.append("0")                 # Flag para usar o dataset de imgs do Google(somente treino, padrao 0)
            cmd.append(inputFolder)         # Pasta para output do dataset gerado
            cmd.append("detection")
            cmd.append("false")
            



            if debug[0]:
                messages.addMessage(" ".join(cmd))
            else:
                # messages.addMessage(" ".join(cmd))
                # preProcess = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                preProcess = subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                preProcess.wait()


                #messages.addMessage("PREP OUT")
                #for line in preProcess.stdout:
                   #messages.addMessage(line)
                #messages.addMessage("PREP ERROR")
                #for line in preProcess.stderr:
                   #messages.addMessage(line)

            # Fine Tunning
            cmd = []
            cmd.append(pythonCmd)
            cmd.append(netCmd)                                  # Script da cnn (abaixo usando como padrao a Faster)
            ## Parametros
            cmd.append('--dir_path')
            cmd.append(inputFolder)                             # Pasta para output do dataset gerado
            cmd.append('--imdb')                   # Flag para definir o uso do dataset do TCU para treino
            cmd.append('tcu_trainval')
            cmd.append('--imdbval')                    # Flag para definir o uso do dataset do TCU para validacao
            cmd.append('tcu_test')
            cmd.append('--iters')                         # Numero total de iteracoes
            cmd.append('20000')
            cmd.append('--cfg')      # Configuracao padrao da rede VGG16
            cmd.append(os.path.join(ROOT_PATH, 'faster\\tf-faster-rcnn\\experiments\\cfgs\\vgg16.yml'))
            cmd.append('--net')                           # Rede VGG16
            cmd.append('vgg16')
            cmd.append('--da')                             # Flag para definir uso de Data Augmentation
            cmd.append('True')
            cmd.append('--set')      # Variáveis importantes do Faster
            cmd.append('ANCHOR_SCALES')
            cmd.append('[8,16,32]')
            cmd.append('ANCHOR_RATIOS')
            cmd.append('[0.5,1,2]')
            cmd.append('TRAIN.STEPSIZE')
            cmd.append('[10000]')

            if debug[1]:
                messages.addMessage(" ".join(cmd))
            else:
                # messages.addMessage(" ".join(cmd))
                infProcess = subprocess.call(cmd)
                # infProcess = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # f_out = open('C:\\Users\\pedro\\Documents\\ArcGIS\\out.txt', 'w')
                # f_err = open('C:\\Users\\pedro\\Documents\\ArcGIS\\err.txt', 'w')
                # infProcess = subprocess.Popen(cmd, shell=False, stdout=f_out, stderr=f_err)
                # infProcess.wait()
                # for file in os.listdir(os.path.join(ROOT_PATH, "faster\\tf-faster-rcnn\\data\\cache\\")):
                #     file_path = os.path.join(ROOT_PATH, "faster\\tf-faster-rcnn\\data\\cache\\", file)
                #     if os.path.isfile(file_path):
                #         os.unlink(file_path)  # remove cache from faster

                #messages.addMessage(" ".join(cmd))
                #messages.addMessage("FASTER OUT")
                #for line in infProcess.stdout:
                   #messages.addMessage(line)
                #messages.addMessage("FASTER ERROR")
                #for line in infProcess.stderr:
                   #messages.addMessage(line)

            # Pos-process
            cmd = []
            cmd.append(pythonCmd)
            cmd.append(posCmd)             # Script de posprocess
            ## Parametros
            cmd.append(",".join(imgfiles))  # Lista de imagens
            cmd.append(inputFolder)         # Pasta para output do dataset gerado
            cmd.append(os.path.join(outputFolder, outputShape))        # Pasta padrao que recebe as saidas(shapefiles e predicoes em img)
            cmd.append("448")               # Tamanho dos crops a ser gerados. 448x448
            cmd.append("0")                 # Flag para indicar se iremos avaliar o resultado usando o shapefile original de entrada
            cmd.append("_")                 # Se a flag anterior for 1 (true), aqui deve passar o path para o shapefile com as marcacoes originais para comapracao e avaliacao


            if debug[0]:
                messages.addMessage(" ".join(cmd))
            else:
                # messages.addMessage(" ".join(cmd))
                preProcess = subprocess.call(cmd)
                # preProcess = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # preProcess.wait()


                #messages.addMessage("POSP OUT")
                #for line in preProcess.stdout:
                   #messages.addMessage(line)
                #messages.addMessage("POSP ERROR")
                #for line in preProcess.stderr:
                   #messages.addMessage(line)

            ## Mostrar shapefile gerado no mapa atual aberto
            outputShapePath = os.path.join(os.path.abspath(outputFolder),outputShape)
            if os.path.exists(outputShapePath):
                arcpy.MakeFeatureLayer_management(outputShapePath, parameters[3].valueAsText)
                mxd = arcpy.mapping.MapDocument('current')
                df = arcpy.mapping.ListDataFrames(mxd)[0]

                layer = arcpy.mapping.Layer(parameters[3].valueAsText)
                arcpy.mapping.AddLayer(df, layer, "AUTO_ARRANGE")
                arcpy.RefreshActiveView()

            shutil.rmtree(inputFolder)

        return

class AnalyzeImgsDet(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Analyze Images (Bridge Detection)"
        self.description = "Tool for analyze images and find railway bridges."
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
        # cmd = 'mode 50, 5'
        # os.system(cmd)
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

        #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        #os.environ["CUDA_VISIBLE_DEVICES"]="0"

        #messages.addMessage(os.environ)
        checkShapeCmd = os.path.join(ROOT_PATH, "utils\\checkShapeFile.py")
        prepCmd = os.path.join(ROOT_PATH, "utils\\createDatasets.py") # Script de preprocessamento dos dados para a Faster
        netCmd = os.path.join(ROOT_PATH, "faster\\tf-faster-rcnn\\tools\\test_net.py")
        posCmd = os.path.join(ROOT_PATH, "faster\\handle_shapefile.py") # Script para gerar o shapefile final
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


        # path_variables = os.environ['path']
        # path_variables = path_variables.split(';')
        # indices_path = [i for i, s in enumerate(path_variables) if 'ython352' in s]
        # if len(indices_path)==0:
        #     messages.addMessage("\n*****\n***You must have Python 3.5.2 installed to run the plugin.***\n*****\n")
        #     return
        # path_python = path_variables[indices_path[0]]
        # pythonCmd = os.path.join(path_python,'python.exe')



        # pythonCmd = os.path.join(os.path.dirname(sys.executable),'python.exe')

        # rlwshapefile = "rlwshp_placeholder"
        # outputFolder = "tcuAnalyzes"
        # if not os.path.exists(outputFolder):
        #     os.mkdir(outputFolder)
        #     os.chmod(outputFolder, 7777)

        # Nao precisa definiir o modelo, o mask trata internamente o melhor modelo pra carregar
        # modelWeights = os.path.join(ROOT_PATH, "faster\\tf-faster-rcnn\\output\\vgg16\\tcu_trainval\\default\\")

        # Imagens
        imgfiles = []
        for raster in parameters[0].values:
            d = arcpy.Describe(raster)
            if sameMachine:
                imgfiles.append(d.catalogPath)
            else:
                imgfiles.append(d.catalogPath.replace('\\', '/'))

        # Railway Shapefile
        d = arcpy.Describe(parameters[1].value)
        rlwShpfilePath = d.catalogPath
        #messages.addMessage(d.catalogPath)

        outputShape = parameters[2].valueAsText + '.shp'
        outputFolder = parameters[3].valueAsText

        modelWeights = os.path.join(ROOT_PATH, "faster\\tf-faster-rcnn\\output\\vgg16\\tcu_trainval\\default\\vgg16_faster_rcnn_iter_10000.ckpt")

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
                # if output != 'MULTILINESTRING':
                #     messages.addMessage("\n*****\n***Check the railway shapefile, it seems it contains more than lines.***\n*****\n")
                #     return

            # Preprocess
            cmd = []
            inputFolder = tempfile.mkdtemp()
            cmd.append(pythonCmd)
            cmd.append(prepCmd)             # Script de preprocess
            ## Parametros
            cmd.append(",".join(imgfiles))  # Lista de imagens
            cmd.append(rlwShpfilePath)      # Shapefile da ferrovia, usado como guia para criar os crops
            cmd.append("_")     # Shapefile das marcacoes de ponte
            cmd.append("448")               # Tamanho dos crops a ser gerados. 448x448
            cmd.append("test")             # Flag para geracao do dataset (treino ou teste)
            #cmd.append("0")                 # Flag para usar o dataset de imgs do Google(somente treino, padrao 0)
            cmd.append(inputFolder)         # Pasta para output do dataset gerado
            cmd.append("detection")
            cmd.append("false")
            # messages.addMessage(" message  ")
            # messages.addMessage(" ".join(cmd))
            # messages.addMessage(" message  ")
            if debug[0]:
                messages.addMessage(" ".join(cmd))
            else:
                #messages.addMessage("Command:")
                #messages.addMessage(" ".join(cmd))
                #preProcess = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                #preProcess.wait()
                subprocess.call(cmd)

                # #messages.addMessage("PREP OUT")
                # for line in preProcess.stdout:
                #    #messages.addMessage(line)
                # #messages.addMessage("PREP ERROR")
                # for line in preProcess.stderr:
                #    #messages.addMessage(line)

            # Fine Tunning
            cmd = []
            cmd.append(pythonCmd)
            cmd.append(netCmd)                                  # Script da cnn (abaixo usando como padrao a Faster)
            ## Parametros
            cmd.append('--dir_path')
            cmd.append(inputFolder)                             # Pasta para output do dataset gerado
            cmd.append('--imdb')                       # Flag para definir o uso do dataset do TCU para teste
            cmd.append('tcu_test')
            cmd.append('--model')
            cmd.append(modelWeights)                            # Numero total de iteracoes
            cmd.append('--cfg')      # Configuracao padrao da rede VGG16
            cmd.append(os.path.join(ROOT_PATH, 'faster\\tf-faster-rcnn\\experiments\\cfgs\\vgg16.yml'))
            cmd.append('--net')
            cmd.append('vgg16')                           # Rede VGG16
            cmd.append('--set')      # Variáveis importantes do Faster
            cmd.append('ANCHOR_SCALES')
            cmd.append('[8,16,32]')
            cmd.append('ANCHOR_RATIOS')
            cmd.append('[0.5,1,2]')

            if debug[1]:
                messages.addMessage(" ".join(cmd))
            else:
                subprocess.call(cmd)
                # infProcess = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # infProcess.wait()

                # #messages.addMessage("Command:")
                # #messages.addMessage(" ".join(cmd))
                # #messages.addMessage("FASTER OUT")
                # for line in infProcess.stdout:
                #    #messages.addMessage(line)
                # #messages.addMessage("FASTER ERROR")
                # for line in infProcess.stderr:
                #    #messages.addMessage(line)

            # Pos-process
            cmd = []
            #inputFolder = tempfile.mkdtemp()
            cmd.append(pythonCmd)
            cmd.append(posCmd)              # Script de posprocess
            ## Parametros
            cmd.append(",".join(imgfiles))  # Lista de imagens
            cmd.append(inputFolder)         # Pasta para output do dataset gerado
            cmd.append(os.path.join(outputFolder, outputShape))        # Pasta padrao que recebe as saidas(shapefiles e predicoes em img)
            cmd.append("448")               # Tamanho dos crops a ser gerados. 448x448
            cmd.append("0")                 # Flag para indicar se iremos avaliar o resultado usando o shapefile original de entrada
            cmd.append("_")                 # Se a flag anterior for 1 (true), aqui deve passar o path para o shapefile com as marcacoes originais para comapracao e avaliacao


            if debug[0]:
                messages.addMessage(" ".join(cmd))
            else:
                subprocess.call(cmd)
                # #messages.addMessage("Command:")
                # #messages.addMessage(" ".join(cmd))
                # preProcess = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # preProcess.wait()


                # #messages.addMessage("POSP OUT")
                # for line in preProcess.stdout:
                #    #messages.addMessage(line)
                # #messages.addMessage("POSP ERROR")
                # for line in preProcess.stderr:
                #    #messages.addMessage(line)

            ## Mostrar shapefile gerado no mapa atual aberto
            outputShapePath = os.path.join(os.path.abspath(outputFolder),outputShape)
            if os.path.exists(outputShapePath):
                arcpy.MakeFeatureLayer_management(outputShapePath, parameters[2].valueAsText)
                mxd = arcpy.mapping.MapDocument('current')
                df = arcpy.mapping.ListDataFrames(mxd)[0]

                layer = arcpy.mapping.Layer(parameters[2].valueAsText)
                arcpy.mapping.AddLayer(df, layer, "AUTO_ARRANGE")
                arcpy.RefreshActiveView()
            else:
                messages.addMessage("No bridges found.")

            shutil.rmtree(inputFolder)
        return

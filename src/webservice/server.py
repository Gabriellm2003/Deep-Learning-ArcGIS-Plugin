#!/usr/bin/python
from http.server import BaseHTTPRequestHandler,HTTPServer
from os import curdir, sep
import io, os, shutil
import sys
import cgi
import tempfile
from zipfile import ZipFile
import subprocess
import re
from subprocess import Popen, PIPE


try:
    PORT_NUMBER = int(sys.argv[1])
except:
    PORT_NUMBER = 8080



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




#This class will handles any incoming request from
#the browser 
class myHandler(BaseHTTPRequestHandler):
    
    def okReport(self, outfile):
        htmlf = open('template.html', 'r')
        html = htmlf.read()
        htmlf.close()

        message = '<div class="card border-info mb-3">'
        message += '<div class="card-header bg-success text-white"><h1>Request Resolved</h1></div>'
        message += '<div class="card-body">'
        message += '<p class="lead">Your request seems to have ended fine :D</p>'
        # message += '<p><mark><em>It\'s gonna take a while to process. After the server finish you can check if your shapefile is\
        #            in the tab <strong>Generated Shapefile</strong></em></mark></p>'
        # message += '<p>The server is now gonna run the task you asked in the following images:</p>'
        
        message += '<p>You can check if everything turned out okay, by going in the tab <strong>Generated Shapefile</strong> \
                    and looking for your shapefile named <strong>' + outfile + '</strong></p>.' 

        # if len(imglist) > 0:
        #     message += '<h4>Image files:</h4>'
        # message += '<ul>'
        # for file in imglist:
        #     message += '<li>' + file + '</li>'
        # message += '</ul>'

        message += '</div></div>'

        htmls = html.split('<div class="col-md-8">')
        response = htmls[0] + '<div class="col-md-8">' + message + htmls[1]

        return response


    def errorReport(self, nonimgs, exrlw, rlwshp, exinshp, inputshp, inshprequirement):
        htmlf = open('template.html', 'r')
        html = htmlf.read()
        htmlf.close()

        card = '<div class="card border-danger mb-3">'
        card += '<div class="card-header bg-danger text-white"><h1>Error in input request</h1></div>'
        card += '<div class="card-body">'
        card += '<p class="lead">It\'s seems that some files doesn\'t exists in the server.</p>'
        card += '<p>Please check the <strong>path</strong> for following files. If the path is correct check the <strong>permission of access</strong> of those files.</p>'
        
        if len(nonimgs) > 0:
            card += '<h4>Image files:</h4>'
        card += '<ul>'
        for file in nonimgs:
            card += '<li>' + file + '</li>'
        card += '</ul>'

        if not exrlw:
            card += '<h4>Railway shapefile:</h4>'
            card += '<ul><li>' + rlwshp + '</li></ul>'

        if inshprequirement and not exinshp:
            card += '<h4>ROI annotation shapefile:</h4>'
            card += '<ul><li>' + inputshp + '</li></ul>'

        card += '</div></div>'
        htmls = html.split('<div class="col-md-8">')
        response = htmls[0] + '<div class="col-md-8">' + card + htmls[1]

        return response


    def folderListing(self, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)

        htmlf = open('template.html', 'r')
        html = htmlf.read()
        htmlf.close()
        table = '<table class="table table-hover table-striped">'
        table += '<thead class="thead-dark"> <tr> <th scope="col"> Filename </th> <th> </th> </tr> </thead>'
        table += '<tbody>'

        ext = ['.dbf', '.prj', '.shx', '.shp']

        
        for file in os.listdir(folder):
            tr = '<tr>'
            if '.shp' in file:
                td = '<td>'
                base = os.path.splitext(file)[0]
                valid = True
                for e in ext:
                    if not os.path.exists(os.path.join(folder, base + e)):
                        valid = False
                        break

                if not valid:
                    continue

                # If don't exist zip file create a new one to contain all files 
                # needed for a useful shapefile
                if not os.path.exists(os.path.join(folder, base + '.zip')):
                    with ZipFile(os.path.join(folder, base + '.zip'), 'w') as zipf:
                        for e in ext:
                            zipf.write(os.path.join(folder, base + e))

                        zipf.close()

                td += base
                td += '</td>'

                td += '<td> <a href="' + os.path.join(folder, base + '.zip') + '" download>'
                td += 'Download file </a></td>'

                tr += td
                tr += '</tr>'

                table += tr

        table += '</tbody>'
        table += '</table>'

        htmls = html.split('<div class="col-md-8">')
        response = htmls[0] + '<div class="col-md-8">' + table + htmls[1]

        return response

    def inputRequestHandler(self, form):

        SRC = os.path.abspath(os.path.split(os.path.dirname(__file__))[0])
        SRC = os.path.abspath('..')
        # pythoncmd = os.path.join('C:\\Python352','python.exe')
        pythoncmd = ''
        path_variables = os.environ['path']
        path_variables = path_variables.split(';')
        candidates = [a for a in path_variables if os.path.isfile(a+"\\python.exe")]
        regex = b".*\(major=(\d), minor=(\d), micro=(\d)"
        foundPython = False
        for i in candidates:
            ret = subprocess.check_output(i + '\\python.exe -c "import sys; print(sys.version_info)"')
            print(" ret: ", ret)
            m = re.search(regex, ret)
            if int(m.group(1)) == 3 and int(m.group(2)) == 5 and int(m.group(3)) == 2:
                # print "esse eh o python 3.5.2"
                pythoncmd = os.path.join(i,'python.exe')
                foundPython = True
        if not foundPython:
            print("\n*****\n***You must have Python 3.5.2 installed to run the plugin.***\n*****\n")
            return
        prepcmd = os.path.join(SRC, 'utils\\createDatasets.py')
        mrcnncmd = os.path.join(SRC, 'mrcnn\\tcu_model\\erosion_detection.py')
        fastercmd = os.path.join(SRC, 'faster\\tf-faster-rcnn\\tools\\trainval_net.py')
        testfastercmd = os.path.join(SRC, "faster\\tf-faster-rcnn\\tools\\test_net.py")
        fasterposcmd = os.path.join(SRC, 'faster\\handle_shapefile.py')
        mrcnnweightspath = os.path.join(SRC, 'mrcnn\\tcu_model\\model1')
        fasterweightspath = os.path.join(SRC, 'faster\\tf-faster-rcnn\\output\\vgg16\\tcu_trainval\\default\\vgg16_faster_rcnn_iter_10000.ckpt')
        imgsplitCmd = os.path.join(SRC, "utils\\imgSpliter.py")

        cropsize = '448'
        outputfolder = 'genshps'



        if not os.path.exists(outputfolder):
            os.mkdir(outputfolder)

        imglist = form.getvalue("imglist")
        rlwshp  = form.getvalue("rlwshp")
        task    = form.getlist("task")
        target = form.getlist("target")
        # target2 = form.getlist("target2")
        outname1 = form.getvalue("outname1")
        outname2 = form.getvalue("outname2")
        inputshp = form.getvalue("inputshp")

        # Check image existence
        nonimgs = []
        for img in imglist.split(","):
            if not os.path.exists(img):
                nonimgs.append(img)

        # Check shapefiles
        exrlw = os.path.exists(rlwshp)
        exinshp = os.path.exists(inputshp)

        inshprequirement = task[0] == "finetune"

        # Error: some files don't exist
        if (len(nonimgs) > 0) or (not exrlw) or (not exinshp and inshprequirement):
            self.send_response(400)
            self.send_header('Content-type','text/html')
            self.end_headers()
            r = self.errorReport(nonimgs, exrlw, rlwshp, exinshp, inputshp, inshprequirement)
            self.wfile.write(bytes(r, 'utf-8'))
            return

        # Seems fine
        else:
            # response = self.okReport(imglist.split(','))
            response = self.okReport(outname1.replace('.shp', ''))
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(bytes(response, 'utf-8'))

        process = 'test' if "analyze" in task else 'train'
        print ("TASK: " + str(task))
        print ("PROCESS: " + str(process))
        featureshp = inputshp if process == 'train' else '_'
        outputshape = outname1 if '.shp' in outname1 else outname1 + '.shp'


        if 'erosion' in target:
            task = 'segmentation'
        else:
            task = 'detection'


        print ("TARGET = " + str(target))
        print ("TASK = " + str(task))
        # Net stuff
        if process == 'test':
            if 'erosion' in target:
                print (imglist)
                # Erosion segmentation task
                cmd = []
                cmd.append(pythoncmd)
                cmd.append(imgsplitCmd)
                cmd.append(imglist)
                p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                output, err = p.communicate(b"")
                rc = p.returncode
                # print('outputtttttttttttttttttt: ', output)
                # print('errrrrrrrrrrrrrrrrrrrrrr: ',err)
                imgs = eval(output)
                outshapes = []

                for num, file in enumerate(imgs):
                    # Preprocess
                    cmd = []
                    inputfolder = tempfile.mkdtemp()
                    cmd.append(pythoncmd)   # Python command 
                    cmd.append(prepcmd)     # Preprocess script (createDataset.py) path
                    cmd.append(file)     # Image list to process
                    cmd.append(rlwshp)      # Railway shapefile path
                    cmd.append(rlwshp)  # Annotations shapefile path (if is the case)
                    cmd.append(cropsize)    # Crop size 
                    cmd.append(process)     # Process (train[finetune]|test)
                    cmd.append(inputfolder) # Temporary folder where crops will be saved
                    cmd.append(task)        # Bridge detection or Erosion Segmentation
                    cmd.append('false')     # createDataset paramenter(use or not the railway as feature)
                    print (cmd)
                    preProcess = subprocess.call(cmd)

                    cmd = []
                    cmd.append(pythoncmd)
                    cmd.append(mrcnncmd)        # Script da cnn (abaixo usando como padrao a MaskRCNN)
                    cmd.append("testing")       # Procedimento que a rede ira executar
                    cmd.append("_")             # Pasta com as imgs de treino
                    cmd.append("_")             # Pasta com as imgs de validacao
                    cmd.append(inputfolder)     # Pasta com as imgs para teste
                    cmd.append(outputfolder)    # Pasta padrao que recebe as saidas(shapefiles e predicoes em img)
                    cmd.append("0")             # Learning rate
                    cmd.append("[0,0,0]")       # Epoch numbers
                    cmd.append(mrcnnweightspath) # Pesos do modelo treinado
                    cmd.append("True")          # Criar um mapa de predicao completo(por padrao o teste ja realiza)
                    cmd.append(outputshape.replace('.shp', '') + '_' + str(num))  # Nome shapefile de output
                    outshapes.append(os.path.join(outputfolder, outputshape.replace('.shp', '') + '_' + str(num) ))


                    print (cmd)
                    infProcess = subprocess.call(cmd)
                    # Delete tempfolder 
                    shutil.rmtree(inputfolder)

                cmd = []
                cmd.append(pythoncmd)
                cmd.append(os.path.join(SRC,'mrcnn\\tcu_model','ogrmerge.py'))
                cmd.append('-o')
                cmd.append(os.path.join(outputfolder,outputshape))
                #messages.addMessage("OUTSHAPES = " + str(outshapes))
                print (outshapes)
                for shp in outshapes:
                   cmd.append(shp + '.shp')
                cmd.append('-single')
                print (cmd)
                #messages.addMessage(" ".join(cmd))
                merge = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                merge.wait()

                # Apagar os shapefiles usados uniao
                # driver = ogr.GetDriverByName("ESRI Shapefile")
                for shp in outshapes:
                    nameShp = shp
                    os.remove(nameShp+'.shp')
                    os.remove(nameShp+'.dbf')
                    os.remove(nameShp+'.prj')
                    os.remove(nameShp+'.shx')
                for num, file in enumerate(imgs):
                    if '_splitpart' in file:
                        os.remove(file)                        
                        try:
                            nameFile = file.replace('.img', '')
                        except:
                            pass
                        try:
                            os.remove(nameFile+'.ige')
                        except:
                            pass

                removeEmptyFolders(outputfolder, False)

            if 'bridge' in target:
                # Bridge detection task
                
                # Preprocess
                cmd = []
                inputfolder = tempfile.mkdtemp()
                cmd.append(pythoncmd)   # Python command 
                cmd.append(prepcmd)     # Preprocess script (createDataset.py) path
                cmd.append(imglist)     # Image list to process
                cmd.append(rlwshp)      # Railway shapefile path
                cmd.append(featureshp)  # Annotations shapefile path (if is the case)
                cmd.append(cropsize)    # Crop size 
                cmd.append(process)     # Process (train[finetune]|test)
                cmd.append(inputfolder) # Temporary folder where crops will be saved
                cmd.append(task)        # Bridge detection or Erosion Segmentation
                cmd.append('false')     # createDataset paramenter(use or not the railway as feature)
                print (" ".join(cmd))
                preProcess = subprocess.call(cmd)

                # Bridge detection net
                cmd = []
                cmd.append(pythoncmd)
                cmd.append(testfastercmd)           # Script da cnn (abaixo usando como padrao a Faster)
                ## Parametros
                cmd.append('--dir_path')
                cmd.append(inputfolder)         # Pasta para output do dataset gerado
                cmd.append('--imdb')            # Flag para definir o uso do dataset do TCU para teste
                cmd.append('tcu_test')
                cmd.append('--model')
                cmd.append(fasterweightspath)   # Pasta com os pesos do modelo
                cmd.append('--cfg')             # Configuracao padrao da rede VGG16
                cmd.append(os.path.join(SRC, 'faster\\tf-faster-rcnn\\experiments\\cfgs\\vgg16.yml'))
                cmd.append('--net')
                cmd.append('vgg16')             # Rede VGG16
                cmd.append('--set')             # VariÃ¡veis importantes do Faster
                cmd.append('ANCHOR_SCALES')
                cmd.append('[8,16,32]')
                cmd.append('ANCHOR_RATIOS')
                cmd.append('[0.5,1,2]')
                print (" ".join(cmd))
                infProcess = subprocess.call(cmd)

                # Pos-process
                cmd = []
                cmd.append(pythoncmd)
                cmd.append(fasterposcmd)             # Script de posprocess
                ## Parametros
                cmd.append(imglist)  # Lista de imagens
                cmd.append(inputfolder)         # Pasta para output do dataset gerado
                cmd.append(os.path.join(outputfolder, outputshape))        # Pasta padrao que recebe as saidas(shapefiles e predicoes em img)
                cmd.append(cropsize)               # Tamanho dos crops a ser gerados. 448x448
                cmd.append("0")                 # Flag para indicar se iremos avaliar o resultado usando o shapefile original de entrada
                cmd.append("_") 
                print (" ".join(cmd))
                infProcess = subprocess.call(cmd)
                # Delete tempfolder 
                shutil.rmtree(inputfolder)

        # Fine tunning
        else:                


            if 'erosion' in target:
                # Fine Tunning segmentation
                # Preprocess
                cmd = []
                inputfolder = tempfile.mkdtemp()
                cmd.append(pythoncmd)   # Python command 
                cmd.append(prepcmd)     # Preprocess script (createDataset.py) path
                cmd.append(imglist)     # Image list to process
                cmd.append(rlwshp)      # Railway shapefile path
                cmd.append(featureshp)  # Annotations shapefile path (if is the case)
                cmd.append(cropsize)    # Crop size 
                cmd.append(process)     # Process (train[finetune]|test)
                cmd.append(inputfolder) # Temporary folder where crops will be saved
                cmd.append('segmentation')        # Bridge detection or Erosion Segmentation
                cmd.append('false')     # createDataset paramenter(use or not the railway as feature)
                print (cmd)
                preProcess = subprocess.call(cmd)
                cmd = []
                cmd.append(pythoncmd)
                cmd.append(mrcnncmd)        # Script da cnn (abaixo usando como padrao a MaskRCNN)
                cmd.append("fnt")           # Procedimento que a rede ira executar
                cmd.append(os.path.join(inputfolder, 'Train'))      # Pasta com as imgs de treino
                cmd.append(os.path.join(inputfolder, 'Validation')) # Pasta com as imgs de validacao
                cmd.append("_")             # Pasta com as imgs para teste
                cmd.append(outputfolder)    # Pasta padrao que recebe as saidas(shapefiles e predicoes em img)
                cmd.append("0.001")         # Learning rate
                cmd.append("[70]")          # Epoch numbers
                cmd.append(mrcnnweightspath)# Pesos do modelo treinado
                cmd.append("False")         # Criar um mapa de predicao completo(nao usado)
                cmd.append("_")             # Nome do shapefile de saida
                print (cmd)
                # preProcess = subprocess.call(cmd)
            else:
                # Fine Tunning detection
                # Preprocess
                cmd = []
                inputfolder = tempfile.mkdtemp()
                cmd.append(pythoncmd)   # Python command 
                cmd.append(prepcmd)     # Preprocess script (createDataset.py) path
                cmd.append(imglist)     # Image list to process
                cmd.append(rlwshp)      # Railway shapefile path
                cmd.append(featureshp)  # Annotations shapefile path (if is the case)
                cmd.append(cropsize)    # Crop size 
                cmd.append(process)     # Process (train[finetune]|test)
                cmd.append(inputfolder) # Temporary folder where crops will be saved
                cmd.append('detection')        # Bridge detection or Erosion Segmentation
                cmd.append('false')     # createDataset paramenter(use or not the railway as feature)
                print (cmd)
                preProcess = subprocess.call(cmd)
                cmd = []
                cmd.append(pythoncmd)
                cmd.append(fastercmd)                               # Script da cnn (abaixo usando como padrao a Faster)
                ## Parametros
                cmd.append('--dir_path')
                cmd.append(inputfolder)                             # Pasta para output do dataset gerado
                cmd.append('--imdb')                   # Flag para definir o uso do dataset do TCU para treino
                cmd.append('tcu_trainval')
                cmd.append('--imdbval')                    # Flag para definir o uso do dataset do TCU para validacao
                cmd.append('tcu_test')
                cmd.append('--iters')                         # Numero total de iteracoes
                cmd.append('20000')
                cmd.append('--cfg')      # Configuracao padrao da rede VGG16
                cmd.append(os.path.join(SRC, 'faster\\tf-faster-rcnn\\experiments\\cfgs\\vgg16.yml'))
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
                print (cmd)
            infProcess = subprocess.call(cmd)
            removeEmptyFolders(outputfolder, False)
            removeOldModels(outputfolder)

            # Detection creates the shapefile out of the net script
            if 'bridge' in target:
                # Pos-process
                cmd = []
                cmd.append(pythoncmd)
                cmd.append(fasterposcmd)             # Script de posprocess
                ## Parametros
                cmd.append(imglist)  # Lista de imagens
                cmd.append(inputfolder)         # Pasta para output do dataset gerado
                cmd.append(os.path.join(outputfolder, outputshape))        # Pasta padrao que recebe as saidas(shapefiles e predicoes em img)
                cmd.append(cropsize)               # Tamanho dos crops a ser gerados. 448x448
                cmd.append("0")                 # Flag para indicar se iremos avaliar o resultado usando o shapefile original de entrada
                cmd.append("_") 
                print (cmd)
                # infProcess = subprocess.call(cmd)
            # Delete tempfolder 
            shutil.rmtree(inputfolder)

    #Handler for the GET requests
    def do_GET(self):
        if self.path=="/":
            self.path="/index.html"

        if self.path=='/shplist':
            self.send_response(200)
            self.send_header('Content-type','text/html')
            self.end_headers()
            r = self.folderListing('genshps')
            self.wfile.write(bytes(r, 'utf-8'))

            return

        try:
            #Check the file extension required and
            #set the right mime type

            sendReply = False
            if self.path.endswith(".html"):
                mimetype='text/html'
                sendReply = True
            if self.path.endswith(".jpg"):
                mimetype='image/jpg'
                sendReply = True
            if self.path.endswith(".gif"):
                mimetype='image/gif'
                sendReply = True
            if self.path.endswith(".js"):
                mimetype='application/javascript'
                sendReply = True
            if self.path.endswith(".css"):
                mimetype='text/css'
                sendReply = True
            if self.path.endswith(".zip"):
                mimetype='application/zip'
                sendReply = True

            if sendReply == True:
                #Open the static file requested and send it
                f = io.FileIO(curdir + sep + self.path) 
                self.send_response(200)
                self.send_header('Content-type',mimetype)
                self.end_headers()
                self.wfile.write(f.readall())
                f.close()
            return


        except IOError:
            self.send_error(404,'File Not Found: %s' % self.path)

    #Handler for the POST requests
    def do_POST(self):
        if self.path=="/execute":
            form = cgi.FieldStorage(
                fp=self.rfile, 
                headers=self.headers,
                environ={'REQUEST_METHOD':'POST',
                         'CONTENT_TYPE':self.headers['Content-Type'],
            })

            self.inputRequestHandler(form)

            return          


try:
    #Create a web server and define the handler to manage the
    #incoming request
    server = HTTPServer(('', PORT_NUMBER), myHandler)
    #print('Started httpserver on port ' , PORT_NUMBER)
    
    #Wait forever for incoming htto requests
    server.serve_forever()

except KeyboardInterrupt:
    #print('^C received, shutting down the web server')
    server.socket.close()

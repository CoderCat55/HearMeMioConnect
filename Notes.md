# Notes
* we have collected data from partiicpants. Recording was 3 seconds participants started from rest position then perform the gesture then do the rest poisition again.
>>Human participant tests
-Not ekstra data: Deney günü commit edilmiş p2'de 20 tane tsk (teşekkürler) dosyası var her ne kadar her kelimeyi 10 defa kaydetmiş olsak da. Bu dataların nerden ve nasıl gelddiğini bilmiyoruz ama işe yara oldukları için onları silmedik

* We plotted these data from calibration_data folder
>> used deleted_arrays\liste3.py for plot file name extraction in order to use it in excel file

* checked everyfile with our eyes and written down a list where should the numpy arrays be cut to eliminate the rest positions from datas
>>All participant together list: deleted_arrays\croppeddata (1).xlsx
>>problems we observed and suggestions
-Bazı dataların başı veya sonu eksik sebepler:
--myo veri göndermemiş
--katılımcılar hareketi yapmamya erken başlamış
--katılımcılar hareketi yapmaya geç başlamış

-Bazı dataların rest bölümü yok veya ayırt etmesi zor özellikle verinin sondaki hareketinin gesture'un karakteristik özelliği mi yoksa dalgalı bir rest datası mı olduğunu ayırt etmek zor.

-Öneriler:
--Veriler daha uzun bir kayıt süresi boyunca toplanabilir bu sayede gestureların kesilmesi en aza indirilir ve rest kısımlarının ayırt edilmesi kolaylaşır.


* the lists are given to the algorithm for it to copy the files into another folder and cut the rest parts specified by us this creates a new folder with same scturcture as calibration folder but with "segmented" data.
>>the name of the new folder: rows_deleted
>>deleted_arrays\deleterows2.py is used to delete the rows 

* these data would be our training data. we might further optimize it by delting the half files but this will be considered another time
## models
* Gesture Model
>>a multiclass model will be trained on all data except for "rest" datas from all participants

* Rest Model
>>a binary model will be trained 
>>>Rest class: "rest" datas from all participants
>>>Non-rest class : all data except for rest data

* classification logic
>Options
- A non-rest algıladıktan sonra diğer resti algılayana kadar verileri sakla, rest algılanınca verileri sınıflandır. (Restten reste kadar veriler)
- B Non-rest algılandıktan sonra belli bir süre bekler sonra sınıflandırır (belli bir zamanı sınıflandırma)
- C non rest algılandıktan sonra biraz örnek okur, tahmin başarımı düşükse biraz daha örnek okur (busy-wait)

-Şimdi 1 tane rest 1 tane gesturemodelimiz var ama ikisi neredyse aynı anda çalışıyor birbirlerini etkilerler mi? ya da sadece gesture model olsa zaten resti de algılayabilecek eğer aldıladığı gesture rest değilse window sizeını arttırıp (bu çok gereksiz ve kafa karıştırıcı olabilir sanırım neyse yine de yazmış bulundum)

* hyperparameter tuning is essential 
>Rest Model
>Getsure Model
** We first need to define some hyperparameters and different models for Rest and Gesture detection.
provide a list of models and hyper parameters
we need confision matrix,f1 score and accuracy of models.
how should test and train data will be splitted?
cross validation? program should also show the training time for each model 

# realtime implementation
* After we find the best models they will be implemented to real time:
>classification will be another process that is triggered by def command funciton in main.py with startcf and stopcf these functions will set is_running flag to true or false 
>It needs to be parallel to other processes and should eror information and results via "result" queue and our main system should read these continously for debugging purposes (see branch )
>classification process should first attach to the saher memory to get data and should write when it cannot receive data 


* 10.1.2026 ToDO
- rawgrid plotları ekle (yogurt_duzelt_seg_2a branchinden) ok
- yogurt_duzelt_seg_2a branchinden branchinden excel dosyalarını al (delted arraysin içinde) ok
- liste3.py ile rest datalarınında isimlerini al p1 ve p2 ekle hepsine x yaz (rest dataları eksikti ya onun için)  ok
- array silme proogramını düzenle ve çalıştır ok
- hiperparametre tuning programı lazım 
- şunların cevapları lazım : 
    We first need to define some hyperparameters and different models for Rest and Gesture detection.
    provide a list of models and hyper parameters
    we need confision matrix,f1 score and accuracy of models.
    how should test and train data will be splitted?
    cross validation? program should also show the training time for each model 
- modeller olduktan sonra real time implemente edicez onun için classification logic seçip onları implemente edeceğiz.


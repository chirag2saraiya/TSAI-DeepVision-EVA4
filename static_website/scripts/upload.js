function uploadAndClassifyImage(){
  var fileInput = document.getElementById('resnet34FileUpload').files;
  if (!fileInput.length){
    return alert('Please choose a file to upload first.');
  }

  var file = fileInput[0];
  var filename = file.name

  var formData = new FormData();
  formData.append(filename, file);

  console.log(filename);


$.ajax({
      async: true,
      crossDomain: true,
      method: 'POST',
      url: 'https://3njah24gii.execute-api.ap-south-1.amazonaws.com/dev/classify',
      data: formData,
      processData: false,
      contentType: false,
      mimeType: "multipart/form-data",
})
.done(function(response){
  console.log(response);
  document.getElementById('result').textContent = response;
})
.fail(function() {alert ("There was an error while sending prediction request to resnet34 model."); });
};


function uploadAndFaceSwapImage(){
  var fileInput1 = document.getElementById('file1').files;
  var fileInput2 = document.getElementById('file2').files;
  if (!fileInput1.length || !fileInput2.length ){
    return alert('Please Upload both the image');
  }

  var file1 = fileInput1[0];
  var file2 = fileInput2[0];
  var filename1 = file1.name
  var filename2 = file2.name

  var formData = new FormData();
  formData.append(filename1, file1);
  formData.append(filename2, file2);

  console.log(filename1);
  console.log(filename2);

  $.ajax({
        async: true,
        crossDomain: true,
        method: 'POST',
        url: 'https://sybtgzac4d.execute-api.ap-south-1.amazonaws.com/dev/align',
        data: formData,
        processData: false,
        contentType: false,
        mimeType: "multipart/form-data",
  })
  .done(function(response){
    response = JSON.parse(response);

    $('#faceSwapResult').attr('src', 'data:image/jpeg;base64,'+response.img);
    //console.log(response.img);
    //if(response.result == "True"){
    //$('#faceSwapResult').attr('src', 'data:image/jpeg;base64,'+response.img);
    //}else{
  //document.getElementById('errorMsg').textContent = "Please upload Valid Image (Containing only One Face)";
  //}

  })
  .fail(function() {alert ("There was an error while sending request to face swap service."); });
};


function mobilenetUploadAndClassifyImage(){
  var fileInput = document.getElementById('mobileNetFileUpload').files;
  if (!fileInput.length){
    return alert('Please choose a file to upload first.');
  }

  var file = fileInput[0];
  var filename = file.name

  var formData = new FormData();
  formData.append(filename, file);

  console.log(filename);

$.ajax({
      async: true,
      crossDomain: true,
      method: 'POST',
      url: 'https://v1agl77crf.execute-api.ap-south-1.amazonaws.com/dev/classify',
      data: formData,
      processData: false,
      contentType: false,
      mimeType: "multipart/form-data",
})
.done(function(response){
  console.log(response);
  document.getElementById('resultMobileNet').textContent = response;
})
.fail(function() {alert ("There was an error while sending prediction request to MobileNet model."); });
};


function uploadAndAlignFace(){
  var fileInput = document.getElementById('faceFileUpload').files;
  if (!fileInput.length){
    return alert('Please choose a file to upload first.');
  }

  var file = fileInput[0];
  var filename = file.name

  var formData = new FormData();
  formData.append(filename, file);

  console.log(filename);


$.ajax({
      async: true,
      crossDomain: true,
      method: 'POST',
      url: 'https://noe3b7wagh.execute-api.ap-south-1.amazonaws.com/dev/align',
      data: formData,
      processData: false,
      contentType: false,
      mimeType: "multipart/form-data",
})
.done(function(response){
  response = JSON.parse(response);

  //console.log(response.img);
  if(response.result == "True"){
  $('#faceResult').attr('src', 'data:image/jpeg;base64,'+response.img);
  }else{
document.getElementById('errorMsg').textContent = "Please upload Valid Image (Containing only One Face)";
}
//$('#faceResult').attr('src', response);
})
.fail(function() {alert ("There was an error while sending request to Face Alignment service."); });
};

function uploadAndRecogniseFace(){
  var fileInput = document.getElementById('faceRecognitionUpload').files;
  if (!fileInput.length){
    return alert('Please choose a file to upload first.');
  }

  var file = fileInput[0];
  var filename = file.name

  var formData = new FormData();
  formData.append(filename, file);

  console.log(filename);


$.ajax({
      async: true,
      crossDomain: true,
      method: 'POST',
      url: 'https://1z54rldc98.execute-api.ap-south-1.amazonaws.com/dev/recognise',
      data: formData,
      processData: false,
      contentType: false,
      mimeType: "multipart/form-data",
})
.done(function(response){
  console.log(response);
  document.getElementById('faceRecognitionResult').textContent = response;
})
.fail(function() {alert ("There was an error while sending prediction request to resnet34 model."); });
};


function uploadAndEstimatePose(){
  var fileInput = document.getElementById('poseEstimateFileUpload').files;
  if (!fileInput.length){
    return alert('Please choose a file to upload first.');
  }

  var file = fileInput[0];
  var filename = file.name

  var formData = new FormData();
  formData.append(filename, file);

  console.log(filename);


$.ajax({
      async: false,
      crossDomain: true,
      method: 'POST',
      url: 'https://bg6deyk7fg.execute-api.ap-south-1.amazonaws.com/dev/estimate_pose',
      data: formData,
      processData: false,
      contentType: false,
      mimeType: "multipart/form-data",
})
.done(function(response){
  response = JSON.parse(response);
  $('#poseEstimateResult').attr('src', 'data:image/jpeg;base64,'+response.img);
//$('#faceResult').attr('src', response);
})
.fail(function() {alert ("There was an error while sending request to Human Pose Estimation service."); });
};

function generateCarImage(){

$.ajax({
      async: false,
      crossDomain: true,
      method: 'GET',
      url: 'https://rum1owumx2.execute-api.ap-south-1.amazonaws.com/dev/generate_image',
      processData: false,
      contentType: false,
})
.done(function(response){
  //response = JSON.parse(response);
  $('#ganResult').attr('src', 'data:image/jpeg;base64,'+response.img);
//$('#faceResult').attr('src', response);
})
.fail(function() {alert ("There was an error while sending request to GAN service."); });
};

function uploadAndGenerateImage(){
  var fileInput = document.getElementById('vaeImageUpload').files;
  if (!fileInput.length){
    return alert('Please choose a file to upload first.');
  }

  var file = fileInput[0];
  var filename = file.name

  var formData = new FormData();
  formData.append(filename, file);

  console.log(filename);


$.ajax({
      async: false,
      crossDomain: true,
      method: 'POST',
      url: ' https://5breslh7il.execute-api.ap-south-1.amazonaws.com/dev/generate_image',
      data: formData,
      processData: false,
      contentType: false,
      mimeType: "multipart/form-data",
})
.done(function(response){
  response = JSON.parse(response);
  $('#vaeImageResult').attr('src', 'data:image/jpeg;base64,'+response.img);
//$('#faceResult').attr('src', response);
})
.fail(function() {alert ("There was an error while sending request to VAE Car Image Generation service."); });
};

function uploadAndGenerateSRImage(){
  var fileInput = document.getElementById('srGANImageUpload').files;
  if (!fileInput.length){
    return alert('Please choose a file to upload first.');
  }

  var file = fileInput[0];
  var filename = file.name

  var formData = new FormData();
  formData.append(filename, file);

  console.log(filename);


$.ajax({
      async: false,
      crossDomain: true,
      method: 'POST',
      url: ' https://r0kbgz1xi0.execute-api.ap-south-1.amazonaws.com/dev/get_sr_image',
      data: formData,
      processData: false,
      contentType: false,
      mimeType: "multipart/form-data",
})
.done(function(response){
  response = JSON.parse(response);
  $('#srGANImageResult').attr('src', 'data:image/jpeg;base64,'+response.img);
//$('#faceResult').attr('src', response);
})
.fail(function() {alert ("There was an error while sending request to VAE Car Image Generation service."); });
};



function uploadAndStyleTransferImage(){
  var fileInput1 = document.getElementById('stylefile').files;
  var fileInput2 = document.getElementById('contentfile').files;
  if (!fileInput1.length || !fileInput2.length ){
    return alert('Please Upload both the image');
  }

  var file1 = fileInput1[0];
  var file2 = fileInput2[0];
  var filename1 = file1.name
  var filename2 = file2.name

  var formData = new FormData();
  formData.append(filename1, file1);
  formData.append(filename2, file2);

  console.log(filename1);
  console.log(filename2);

  $.ajax({
        async: false,
        crossDomain: true,
        method: 'POST',
        url: 'https://uteya23kh0.execute-api.ap-south-1.amazonaws.com/dev/classify',
        data: formData,
        processData: false,
        contentType: false,
        mimeType: "multipart/form-data",
  })
  .done(function(response){
    response = JSON.parse(response);

    $('#styleTransferResult').attr('src', 'data:image/jpeg;base64,'+response.img);
   
  })
  .fail(function() {alert ("There was an error while sending request to face swap service."); });
};

function germanToEnglishTranslation(){

  var germanText =  document.getElementById('inputText').value;
  console.log(germanText);
  
  var textData = JSON.stringify({ "inText": germanText}); 
  console.log(textData);


$.ajax({
      async: false,
      crossDomain: true,
      method: 'POST',
      url: 'https://4zszglnlx2.execute-api.ap-south-1.amazonaws.com/dev/translate',
      data: textData,
      processData: false,
      contentType: false,
      mimeType: "application/json",
})
.done(function(response){
  console.log(response);
  document.getElementById('translationResult').textContent = response["output"];
})
.fail(function() {alert ("There was an error while sending request to Image Captioning Service."); });
};



function uploadAndGenerateCaption(){
  var fileInput = document.getElementById('imageCaptionFileUpload').files;
  if (!fileInput.length){
    return alert('Please choose a file to upload first.');
  }

  var file = fileInput[0];
  var filename = file.name

  var formData = new FormData();
  formData.append(filename, file);

  console.log(filename);


$.ajax({
      async: false,
      crossDomain: true,
      method: 'POST',
      url: 'https://1qtjnhjjti.execute-api.ap-south-1.amazonaws.com/dev/get_caption',
      data: formData,
      processData: false,
      contentType: false,
      mimeType: "multipart/form-data",
})
.done(function(response){
  console.log(response);
  document.getElementById('imageCaptionResult').textContent = response;
})
.fail(function() {alert ("There was an error while sending request to Image Captioning Service."); });
};



$('#btnFaceUpload').click(uploadAndAlignFace);
$('#btnFaceRecognitionUpload').click(uploadAndRecogniseFace);
$('#btnResNetUpload').click(uploadAndClassifyImage);
$('#btnMobileNetUpload').click(mobilenetUploadAndClassifyImage);
$('#btnFaceSwap').click(uploadAndFaceSwapImage);
$('#btnPoseEstimateFileUpload').click(uploadAndEstimatePose);
$('#btnGenerateCarImage').click(generateCarImage);
$('#btnVaeImageGenerate').click(uploadAndGenerateImage);
$('#btnSRImageGenerate').click(uploadAndGenerateImage);
$('#btnStyleTransfer').click(uploadAndStyleTransferImage);
$('#btnTranslate').click(germanToEnglishTranslation);
$('#btnImgCaptionUpload').click(uploadAndGenerateCaption);



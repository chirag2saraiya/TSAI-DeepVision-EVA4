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
  var file3;
  var file4;
  var filename1 = file1.name
  var filename2 = file2.name

  var formData1 = new FormData();
  formData1.append(filename1, file1);
  
  // Align first image
  $.ajax({
    async: false,
    crossDomain: true,
    method: 'POST',
    url: 'https://noe3b7wagh.execute-api.ap-south-1.amazonaws.com/dev/align',
    data: formData1,
    processData: false,
    contentType: false,
    mimeType: "multipart/form-data",
})
.done(function(response){
response = JSON.parse(response);
console.log("aligned first image");
$('#faceSwapResult').attr('src', 'data:image/jpeg;base64,'+response.img);
file3 = response.img
    
  })
  .fail(function() {alert ("There was an error while sending request to face swap service."); });


  console.log(file3)
  var formData2 = new FormData();
  formData2.append(filename2, file2);

  // Align first image
  $.ajax({
    async: false,
    crossDomain: true,
    method: 'POST',
    url: 'https://noe3b7wagh.execute-api.ap-south-1.amazonaws.com/dev/align',
    data: formData2,
    processData: false,
    contentType: false,
    mimeType: "multipart/form-data",
})
.done(function(response){
response = JSON.parse(response);
file4 = response.img
console.log("aligned second image");
$('#faceSwapResult').attr('src', 'data:image/jpeg;base64,'+response.img);

  })
  .fail(function() {alert ("There was an error while sending request to face swap service."); });

  //file4 = document.getElementById('faceSwapResult').src;
  console.log(file4)
  var formData = new FormData();
  formData.append(filename1, file3);
  formData.append(filename2, file4);

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
      async: false,
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

function uploadAlignRecognizeFace(){



	console.log("Face Recognition service called");
  var fileInput = document.getElementById('faceRecoFileUpload').files;
  if (!fileInput.length){
    return alert('Please choose a file to upload first.');
  }

  var file = fileInput[0];
  var alignedFile;
  var filename = file.name

  var formData1 = new FormData();
  formData1.append(filename, file);

  console.log(filename);
  
  // Align image
  $.ajax({
    async: false,
    crossDomain: true,
    method: 'POST',
    url: 'https://noe3b7wagh.execute-api.ap-south-1.amazonaws.com/dev/align',
    data: formData1,
    processData: false,
    contentType: false,
    mimeType: "multipart/form-data",
})
.done(function(response){
response = JSON.parse(response);
console.log("aligned first image");
alignedFile = response.img
    
  })
  .fail(function() {alert ("There was an error while sending request to face align service."); });


  console.log(alignedFile)
  
  var formData2 = new FormData();
  formData2.append(filename, alignedFile);
  


$.ajax({
      async: false,
      crossDomain: true,
      method: 'POST',
      url: 'https://z827uu153k.execute-api.ap-south-1.amazonaws.com/dev/recognize',
      data: formData2,
      processData: false,
      contentType: false,
      mimeType: "multipart/form-data",
})
.done(function(response){
  console.log(response);
  document.getElementById('resultFaceRecogition').textContent = response;
})
.fail(function() {alert ("There was an error while sending prediction request to Face Recognition model."); });
};


$('#btnFaceUpload').click(uploadAndAlignFace);
$('#btnResNetUpload').click(uploadAndClassifyImage);
$('#btnMobileNetUpload').click(mobilenetUploadAndClassifyImage);
$('#btnFaceSwap').click(uploadAndFaceSwapImage);
$('#btnFaceRecoUpload').click(uploadAlignRecognizeFace);


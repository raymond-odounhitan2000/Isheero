const GALLERY_ENDPOINT = "http://localhost:8000/images";
const ANNOTATE_ENDPOINT = "http://localhost:8000/annotate";

// Charger les images depuis l'API

fetch(GALLERY_ENDPOINT)
  .then(res => res.json())
  .then(data => {
    const gallery = document.getElementById('image-gallery');
    data.images.forEach(img => {
      const imageElem = document.createElement('img');
      imageElem.src = img.base64;
      imageElem.alt = img.filename;
      gallery.appendChild(imageElem);
    });
  })
  .catch(err => console.error("Erreur chargement images :", err));

function submitImage() {
  const fileInput = document.getElementById('upload-input');
  const file = fileInput.files[0];
  if (!file) return alert("Veuillez choisir une image");

  const formData = new FormData();
  formData.append("file", file);

  fetch(ANNOTATE_ENDPOINT, {
    method: "POST",
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    const section = document.getElementById("result-section");
    section.classList.remove("hidden");

    document.getElementById("annotated-image").src = "http://localhost:8000/" + data.résultats.chemin_image_annotée;
    document.getElementById("description").innerText = "📄 Description : " + data.résultats.description;

    const objectsList = document.getElementById("objects-list");
    objectsList.innerHTML = "";
    data.résultats.détections.objets.forEach(obj => {
      const li = document.createElement('li');
      li.innerText = `🟢 ${obj.classe} (confiance: ${obj.confiance})`;
      objectsList.appendChild(li);
    });
  })
  .catch(err => {
    console.error("Erreur annotation :", err);
    alert("Erreur lors de l'annotation de l'image.");
  });
}

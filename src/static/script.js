const body = document.querySelector("body");
const modal = document.querySelector(".modal");

const openModal = () => {
  modal.classList.add("is-open");
  body.style.overflow = "hidden";
};

window.addEventListener("load", openModal);

document.onkeydown = evt => {
  evt = evt || window.event;
  evt.keyCode === 27 ? closeModal() : false;
};


// The Jupyter widget (anywidget) of the WebGL viewer. The arrays of the scene are
// transferred as binary buffers (DataViews), the view is updated with the scene.

export default {
  render({ model, el }) {
    el.style.display = "block";
    el.style.width = "100%";
    el.style.height = model.get("height");

    const viewer = createViewer(el, model.get("scene"));

    const update = () => viewer.update(model.get("scene"));
    const resize = () => {
      el.style.height = model.get("height");
      viewer.redraw();
    };
    model.on("change:scene", update);
    model.on("change:height", resize);

    return () => {
      model.off("change:scene", update);
      model.off("change:height", resize);
      viewer.dispose();
    };
  },
};

package com.rustbuilder.ui.hints;

import javafx.scene.Node;
import javafx.scene.control.Label;
import javafx.scene.layout.VBox;
import javafx.stage.Popup;

public class ContextHintPopup {

    private static ContextHintPopup instance;
    private final Popup popup;
    private final Label titleLabel;
    private final Label descLabel;
    private Node currentOwner;

    private ContextHintPopup() {
        popup = new Popup();
        popup.setAutoHide(true);
        popup.setHideOnEscape(true);
        // It's important not to consume auto-hiding events in a way that blocks other UI interaction
        popup.setConsumeAutoHidingEvents(false);

        titleLabel = new Label();
        titleLabel.setStyle("-fx-font-weight: bold; -fx-text-fill: #ffffff; -fx-padding: 0 0 4 0;");

        descLabel = new Label();
        descLabel.setWrapText(true);
        descLabel.setMaxWidth(280);
        descLabel.setStyle("-fx-text-fill: #cccccc; -fx-font-size: 12px;");

        VBox box = new VBox(titleLabel, descLabel);
        box.setStyle(
            "-fx-background-color: #2b2b2b; " +
            "-fx-padding: 10 12 10 12; " +
            "-fx-background-radius: 6; " +
            "-fx-border-color: #444444; " +
            "-fx-border-radius: 6; " +
            "-fx-effect: dropshadow(three-pass-box, rgba(0,0,0,0.6), 8, 0, 0, 3);"
        );

        popup.getContent().add(box);
    }

    public static ContextHintPopup getInstance() {
        if (instance == null) {
            instance = new ContextHintPopup();
        }
        return instance;
    }

    public void show(Node owner, String title, String text, double x, double y) {
        if (popup.isShowing()) {
            popup.hide();
        }

        if (owner == null || owner.getScene() == null || owner.getScene().getWindow() == null) {
            return;
        }

        currentOwner = owner;

        boolean hasTitle = (title != null && !title.trim().isEmpty());
        titleLabel.setText(title);
        titleLabel.setVisible(hasTitle);
        titleLabel.setManaged(hasTitle);

        descLabel.setText(text);

        // Add small offset to avoid appearing exactly under the cursor (which could trigger instant mouse exit)
        popup.show(owner.getScene().getWindow(), x + 15, y + 15);
    }

    public void hide() {
        if (popup.isShowing()) {
            popup.hide();
        }
        currentOwner = null;
    }

    public boolean isShowingFor(Node owner) {
        return popup.isShowing() && currentOwner == owner;
    }
}

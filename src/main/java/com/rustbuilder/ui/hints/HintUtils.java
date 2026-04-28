package com.rustbuilder.ui.hints;

import javafx.scene.Node;
import javafx.scene.input.MouseButton;

public class HintUtils {

    /**
     * Attaches a right-click hint popup to a given UI node using a predefined HintKey.
     *
     * @param node    The UI element (e.g., Label, Text) to attach the hint to.
     * @param hintKey The key containing the hint's title and explanation.
     */
    public static void attachHint(Node node, HintKey hintKey) {
        if (node == null || hintKey == null) {
            return;
        }
        attachHint(node, hintKey.getDisplayName(), hintKey.getExplanation());
    }

    /**
     * Attaches a right-click hint popup to a given UI node.
     *
     * @param node        The UI element to attach the hint to.
     * @param title       The title of the hint (optional, can be null).
     * @param explanation The detailed explanation text.
     */
    public static void attachHint(Node node, String title, String explanation) {
        if (node == null || explanation == null || explanation.trim().isEmpty()) {
            return;
        }

        // Hide popup when mouse exits the node
        node.setOnMouseExited(e -> {
            ContextHintPopup popup = ContextHintPopup.getInstance();
            if (popup.isShowingFor(node)) {
                popup.hide();
            }
        });

        // Show popup on secondary click or context menu request
        // Using ContextMenuRequested ensures we capture both right-clicks and keyboard context menu keys
        node.setOnContextMenuRequested(e -> {
            ContextHintPopup.getInstance().show(node, title, explanation, e.getScreenX(), e.getScreenY());
            e.consume(); // Prevent default context menu from appearing if any
        });

        // If the user clicks with the primary button while the popup is open, close it.
        // We use setOnMousePressed to capture it early.
        node.setOnMousePressed(e -> {
            if (e.getButton() == MouseButton.PRIMARY) {
                ContextHintPopup popup = ContextHintPopup.getInstance();
                if (popup.isShowingFor(node)) {
                    popup.hide();
                }
            }
        });
    }
}

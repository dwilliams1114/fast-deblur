package deconvolution;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Cursor;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.KeyEventDispatcher;
import java.awt.KeyboardFocusManager;
import java.awt.RenderingHints;
import java.awt.Toolkit;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.awt.dnd.DropTarget;
import java.awt.dnd.DropTargetDragEvent;
import java.awt.dnd.DropTargetDropEvent;
import java.awt.dnd.DropTargetEvent;
import java.awt.dnd.DropTargetListener;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.ComponentEvent;
import java.awt.event.ComponentListener;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.plugins.jpeg.JPEGImageWriteParam;
import javax.imageio.stream.FileImageOutputStream;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.ButtonGroup;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JRadioButton;
import javax.swing.JSeparator;
import javax.swing.JSpinner;
import javax.swing.SpinnerModel;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.UIManager;
import javax.swing.border.EmptyBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import javax.swing.filechooser.FileNameExtensionFilter;

// Created by Daniel Williams
// Created on April 6, 2019 (copied from PhotoInterface merging program).
// This program was created to test and perfect my fast deconvolution algorithm.
// Last updated on May 6, 2024
/* For OpenGL usage, need to run with VM arguments:
--add-exports java.base/java.lang=ALL-UNNAMED
--add-exports java.desktop/sun.awt=ALL-UNNAMED
--add-exports java.desktop/sun.java2d=ALL-UNNAMED
*/

public class UserInterface {
	private static int X = 900;
	private static int Y = 640;
	
	// This is the image used for previewing
	static BufferedImage previewImage;
	
	// Whether to render the experimental graph after every render
	private static boolean renderExperimentalGraph = false;
	
	static String fileName = null;
	private static String fileDirectory = null;
	
	// This keeps track of where the last file was opened from
	private static String lastFileDirectory = "src/deconvolution";
	
	// This keeps track of where the last file was saved
	private static String lastSaveFileDirectory = "src/deconvolution";
	
	// The status to display at the bottom of the screen
	private static String currentProcessName = "";
	
	// The last time the status was updated
	private static double lastStatusUpdatePercent = 0;
	
	// The amount of zoom to apply 
	private static double previewZoomFactor = 1.0;
	
	// The offset from the center of the preview image
	private static double previewOffsetX = 0;
	private static double previewOffsetY = 0;
	
	// The previous mouse position used for panning the image
	private static int previousMouseX = -1;
	private static int previousMouseY = -1;
	
	// The last kernel used by the deblurring algorithm
	static float[][] lastKernel;
	
	// True if this program is running as a JAR file
	static boolean isPackagedAsJar;
	
	// Various Java Swing window components
	static JFrame frame;
	private static JLabel statusLabel = new JLabel("", SwingConstants.LEFT);
	static JLabel imageInfoLabel = new JLabel("", SwingConstants.RIGHT);
	private static JLabel imageNameLabel;
	private static JProgressBar progressBar;
	private static BufferedImage renderImage = new BufferedImage( // Image onto which graphics are rendered (except OpenGL mode)
			X, Y, BufferedImage.TYPE_INT_RGB);
	private static Graphics2D g = renderImage.createGraphics();	// Graphics used for rendering the image
	private static JLabel renderLabel;			// Used to hold a BufferedImage to render graphics
	private static JPanel interactionFrame;		// Panel used to catch mouse events
	
	public static void main(String[] args) {
		
		// Determine if we are running from inside a JAR file
		isPackagedAsJar = !new File("src").exists();
		
		setupFrame();
		
		/* Measure NRMSE of a processed image
		Algorithms.useOpenCL = true;
		final BufferedImage rawImageBlur = loadImageFromFile(new File("src/deconvolution/Results/JWST 512 64.png"));
		final BufferedImage rawImageOrig = loadImageFromFile(new File("src/deconvolution/Results/JWST 512.png"));
		float[][][] originalBlur = Algorithms.imageToArray(rawImageBlur);
		float[][][] originalOrig = Algorithms.imageToArray(rawImageOrig);
		//float[][][] deblurred = Algorithms.deblurBasic2Switch(originalBlur, 1f, 33.5f, 1, true);
		//float[][][] deblurred = Algorithms.richardsonLucySwitch(originalBlur, 16f, 31, true);
		float[][][] deblurred = Algorithms.sharpenSwitch(originalBlur, 0.6f, 33.5f, true);
		double nrmseBlur = Algorithms.normalizedRootMeanSquareError(originalOrig, originalBlur);
		double nrmseDeblur = Algorithms.normalizedRootMeanSquareError(originalOrig, deblurred);
		print("Blurred: " + nrmseBlur);
		print("Deblur:  " + nrmseDeblur);
		previewImage = Algorithms.arrayToImage(deblurred);
		Algorithms.imageArray = deblurred;
		redrawPreviewImage();
		//*/
		
		/* Process and display an image
		Algorithms.useOpenCL = true;
		final BufferedImage rawImage = loadImageFromFile(new File("src/deconvolution/Results/JWST 512 64.png"));
		float[][][] image1 = Algorithms.imageToArray(rawImage);
		//image1 = Algorithms.fastBlur(image1, 10);
		//image1 = Algorithms.deblurBasic2Switch(image1, 1f, 8.00f, 1, true);
		image1 = Algorithms.richardsonLucySwitch(image1, 64.00f, 8000, true);
		previewImage = Algorithms.arrayToImage(image1);
		Algorithms.imageArray = image1;
		redrawPreviewImage();
		//*/
		
		/*
		final BufferedImage rawImage = loadImageFromFile(new File("src/photoMerger/ImageBlur1.jpg"));
		float[][][] image1 = Algorithms.imageToArray(rawImage);
		// Parameters: image, feather, mult, radius (1.5, 5, 22, 30)
		//image1 = Algorithms.deblurCustom(image1, 100, 200, 22);
		image1 = Algorithms.deblurCustomEnhanced(image1, 100, 70, 5, 3, 1);
		//float[][][] image2 = Algorithms.sharpen(image1, 1.0f, 20);
		BufferedImage newImage = Algorithms.arrayToImage(image1);
		drawImage(newImage);
		renderLabel.repaint();
		//*/
		
		/* Measure NRMSE between two images
		final BufferedImage rawImageBlur = loadImageFromFile(new File("src/deconvolution/Results/Geometry.png"));
		final BufferedImage rawImageOrig = loadImageFromFile(new File("src/deconvolution/Results/FastMethod12.png"));
		float[][][] originalBlur = Algorithms.imageToArray(rawImageBlur);
		float[][][] originalOrig = Algorithms.imageToArray(rawImageOrig);
		double nrmseBlur = Algorithms.normalizedRootMeanSquareError(originalOrig, originalBlur);
		print("NRMSE: " + nrmseBlur);
		System.exit(0);
		//*/
		
		/*
		final BufferedImage rawImageBlur = loadImageFromFile(new File("src/deconvolution/Point Radius 50 2 - Copy.png"));
		float[][][] originalBlur = Algorithms.imageToArray(rawImageBlur);
		float weight = 3.93f;
		float ratio1 = 1.519f;
		float ratio2 = 0.50445f;
		float amountOffset = 2.6407f;
		float[][][] deblurred = Algorithms.deblurBasic2TEST(originalBlur, originalBlur, amountOffset, 12, ratio1, ratio2, weight);
		BufferedImage newImage = Algorithms.arrayToImage(deblurred);
		drawImage(newImage);
		renderLabel.repaint();
		//*/
	}
	
	// Draw this image with the given offset relative to the center of the window
	static void drawImage(final BufferedImage image) {
		g.clearRect(0, 0, X, Y);
		g.drawImage(image,
				(int)(X/2.0 - image.getWidth()/2.0),
				(int)(Y/2.0 - image.getHeight()/2.0), null);
		
		// If there was a kernel used recently, then render it in the corner
		if (lastKernel != null) {
			int cellSize = 150 / lastKernel.length;
			cellSize = Math.min(Math.max(cellSize, 1), 10);
			
			for (int x = 0; x < lastKernel.length; x++) {
				for (int y = 0; y < lastKernel[0].length; y++) {
					int col = (int)(lastKernel[x][y] * 255);
					col %= 256;
					if (col < 0) {
						col += 256;
					}
					g.setColor(new Color(col, col, col));
					g.fillRect(cellSize * x, Y - cellSize * y - cellSize, cellSize, cellSize);
				}
			}
		}
		
		renderLabel.repaint(10);
	}
	
	// Load a an image as a BufferedImage and return it
	private static BufferedImage loadImageFromFile(final File file) {
		try {
			// Load the image into a BufferedImage
			BufferedImage temp = ImageIO.read(file);
			BufferedImage image = new BufferedImage(temp.getWidth(), temp.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
			Graphics2D g = image.createGraphics();
			g.drawImage(temp, 0, 0, null);
			g.dispose();
			
			renderExperimentalGraph = file.getName().toLowerCase().startsWith("test section");
			
			return image;
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return null;
	}
	
	// Load an image into the program and prepare it
	private static void loadImage(final File file) {
		fileName = file.getName();
		fileDirectory = file.getAbsolutePath();
		
		setProcessName("Loaded image " + fileName);
		
		// Show the image filename
		imageNameLabel.setText(fileName);
		
		// Record the most recent directory for convenience
		int lastSlashIndex = file.getAbsolutePath().lastIndexOf('\\');
		if (lastSlashIndex != -1) {
			String folder = file.getAbsolutePath().substring(0, lastSlashIndex);
			lastFileDirectory = folder;
		}
		
		// Load the image
		BufferedImage image = loadImageFromFile(file);
		loadBufferedImage(image);
	}
	
	// Load the given buffered image into the program for viewing and processing
	private static void loadBufferedImage(BufferedImage newImage) {
		previewImage = newImage;
		
		// Convert the BufferedImage to a float array
		Algorithms.imageArray = Algorithms.imageToArray(previewImage);
		
		// Reset the pan and zoom
		previewOffsetX = 0;
		previewOffsetY = 0;
		previewZoomFactor = Math.min(Math.min(
				(double)(X - 10) / Algorithms.imageArray[0].length,
				(double)(Y - 10) / Algorithms.imageArray[0][0].length), 1);
		
		imageInfoLabel.setText(previewImage.getWidth() + "x" + previewImage.getHeight() + " px");

		// Set the transform for the OpenGL canvas just in case we're using it
		DeblurOpenGL.setTransform(previewZoomFactor, -previewOffsetX, previewOffsetY);
		
		// Create a scaled preview version of the image
		redrawPreviewImage();
		
		// Display in OpenGL if applicable
		displayOriginalPreviewOpenGL();
		
		updateProgress(1);
	}
	
	// Set the current process name to be displayed on the screen
	static void setProcessName(final String name) {
		currentProcessName = name;
		updateProgress(0);
	}
	
	// Set the displayed task and progress percentage
	static void updateProgress(final double amount) {
		if (progressBar == null) {
			return;
		}
		
		if ((int)(amount * 100) - (int)(lastStatusUpdatePercent * 100) >= 1 ||
				amount == 1 || amount < lastStatusUpdatePercent) {
			lastStatusUpdatePercent = amount;
			statusLabel.setText(currentProcessName + ":   " + (int)(amount * 100) + "%");
			if (amount == 1) {
				progressBar.setVisible(false);
			} else {
				progressBar.setVisible(true);
				progressBar.setValue((int)(amount * 100));
			}
		}
	}
	
	// Display "Canceled" with no percentage or progress bar
	static void cancelProgress() {
		progressBar.setVisible(false);
		statusLabel.setText("Canceled");
	}
	
	// Render a graph of the color intensity through a cross section of the image
	private static void renderCrossSectionGraph(final BufferedImage image) {
		final int y = image.getHeight()/2;
		g.setColor(Color.WHITE);
		g.fillRect(0, 0, X, (int)(Y*0.2));
		
		// Find the minimum and maximum brightness
		float min = 99999;
		float max = -99999;
		for (int x = 0; x < image.getWidth(); x++) {
			Color col = new Color(image.getRGB(x, y));
			float brightness = (col.getRed() + col.getGreen() + col.getBlue()) / 3.0f;
			min = Math.min(brightness, min);
			max = Math.max(brightness, max);
		}
		
		g.setColor(Color.RED);
		
		float lastY = -1;
		float lastX = -1;
		for (int x = 0; x < image.getWidth(); x++) {
			Color col = new Color(image.getRGB(x, y));
			float brightness = (col.getRed() + col.getGreen() + col.getBlue()) / 3.0f;
			float newY = (brightness - min) / (max - min) * Y * 0.18f + Y * 0.01f;
			float newX = (float)x / image.getWidth() * X;
			
			if (lastX != -1) {
				g.drawLine((int)lastX, (int)lastY, (int)newX, (int)newY);
			}
			
			lastX = newX;
			lastY = newY;
		}
	}
	
	// Create and render the preview image
	static void redrawPreviewImage() {
		if (previewImage == null) {
			return;
		}
		
		if (!Algorithms.useOpenGL) {
			final BufferedImage image = createPreviewImage(
					previewOffsetX, previewOffsetY, previewZoomFactor);
			
			drawImage(image);
		} else {
			
			// Time the execution of the OpenGL process
			final long startTime = System.nanoTime();
			
			DeblurOpenGL.render();
			
			final long endTime = System.nanoTime();
			UserInterface.setProcessName("OpenGL Deblurring (" + ((endTime - startTime) / 1000000) + "ms)");
			UserInterface.updateProgress(1.0);
		}
		
		// If experimental graph is enabled, then render that
		if (renderExperimentalGraph) {
			renderCrossSectionGraph(previewImage);
			renderLabel.repaint();
		}
	}
	
	// Transform, zoom, scale, and return the image being manipulated
	private static BufferedImage createPreviewImage(double zoomCenterX, double zoomCenterY, double zoomFactor) {
		final int width = previewImage.getWidth();
		final int height = previewImage.getHeight();
		
		final BufferedImage newImage = new BufferedImage(X, Y, BufferedImage.TYPE_INT_RGB);
		final Graphics2D g = newImage.createGraphics();
		g.drawImage(previewImage,
				(int)(-zoomCenterX - width/2.0*zoomFactor + newImage.getWidth()/2.0),
				(int)(-zoomCenterY - height/2.0*zoomFactor + newImage.getHeight()/2.0),
				(int)(width * zoomFactor), (int)(height * zoomFactor), null);
		g.dispose();
		
		return newImage;
	}
	
	// Display the original image (not deblurred) using the OpenGL implementation
	static void displayOriginalPreviewOpenGL() {
		if (!Algorithms.useOpenGL) {
			return;
		}
		
		if (previewImage != null) {
			SwingUtilities.invokeLater(new Runnable() {
				public void run() {
					DeblurOpenGL.setDeblurEnabled(false);
					
					// Deblur radius here is ignored
					DeblurOpenGL.setImageParameters(previewImage.getWidth(), previewImage.getHeight(), 12, 1); 
					DeblurOpenGL.setImageToRender(Algorithms.extractByteArray(previewImage));
					DeblurOpenGL.render();
				}
			});
		}
	}
	
	// Remove the existing canvas for rendering, and use the GLCanvas from DeblurOpenGL.java instead.
	private static void enableOpenGLCanvasMode() {
		DeblurOpenGL.initializeOpenGLCanvas();
		
		// Match the size of the GLCanvas to whatever the renderImage was previously
		DeblurOpenGL.glCanvas.setPreferredSize(new Dimension(X, Y));
		
		// Render the image
		displayOriginalPreviewOpenGL();
		
		// Replace the renderLabel with the GLCanvas
		interactionFrame.remove(renderLabel);
		if (!componentContains(interactionFrame, DeblurOpenGL.glCanvas)) {
			interactionFrame.add(DeblurOpenGL.glCanvas, BorderLayout.CENTER);
		}
		
		SwingUtilities.updateComponentTreeUI(interactionFrame);
	}
	
	// Remove the GLCanvas from the frame, and replace it with the normal renderLabel
	private static void disableOpenGLCanvasMode() {
		
		// Replace the GLCanvas with the renderLabel
		if (DeblurOpenGL.glCanvas != null) {
			interactionFrame.remove(DeblurOpenGL.glCanvas);
		}
		
		// Add the normal render label
		if (!componentContains(interactionFrame, renderLabel)) {
			interactionFrame.add(renderLabel);
		}
	}
	
	// Give an error message that images processed using OpenGL cannot be saved or manipulated.
	static void displayOpenGLSaveWarning() {
		
		JOptionPane.showMessageDialog(frame,
				"Cannot manipulate or save images rendered with OpenGL",
				"OpenGL Limitation",
				JOptionPane.ERROR_MESSAGE);
	}
	
	// Return true if the given JComponent contains the given Component
	private static boolean componentContains(JComponent container, Component comp) {
		final Component[] components = container.getComponents();
		for (int i = 0; i < components.length; i++) {
			if (components[i] == comp) {
				return true;
			}
		}
		return false;
	}
	
	// Prompt the user for the deblur radius
	// before playing a video.
	static double promptVideoDeblurRadius() {
		JSpinner spinner = new JSpinner(new SpinnerNumberModel(8.0, 0.25, 256, 0.25));
        int option = JOptionPane.showOptionDialog(frame, spinner,
        		"Enter the blur radius", JOptionPane.OK_CANCEL_OPTION,
        		JOptionPane.QUESTION_MESSAGE, null, null, null);
        if (option == JOptionPane.OK_OPTION) {
            final double blurRadius = (Double)spinner.getValue();
            
            return blurRadius;
            
        } else {
            return -1;
        }
	}
	
	// Initialize the graphic window
	static void setupFrame() {
		
		try {
			UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName()); 
		} catch (Exception e) {}
		
		// Get a desirable window size
		final Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
		X = (int)(screenSize.getWidth() * 0.68 - 150);
		Y = (int)(screenSize.getHeight() * 0.8 - 60);
		
		frame = new JFrame("Fast Deblur Algorithms - Daniel Williams 2024");
		
		renderImage = new BufferedImage(
				X, Y, BufferedImage.TYPE_INT_RGB);
		g = renderImage.createGraphics();
		renderLabel = new JLabel(new ImageIcon(renderImage));
		
		interactionFrame = new JPanel();
		interactionFrame.addMouseWheelListener(new MouseWheelListener() {
			public void mouseWheelMoved(MouseWheelEvent e) {
				
				double mouseX = e.getX() - X/2.0;
				double mouseY = e.getY() - Y/2.0;
				
				if (e.getWheelRotation() < 0) {
					previewZoomFactor *= 1.15;
					
					previewOffsetX += mouseX;
					previewOffsetX *= 1.15;
					previewOffsetX -= mouseX;
					
					previewOffsetY += mouseY;
					previewOffsetY *= 1.15;
					previewOffsetY -= mouseY;
				} else {
					previewZoomFactor /= 1.15;
					
					previewOffsetX += mouseX;
					previewOffsetX /= 1.15;
					previewOffsetX -= mouseX;
					
					previewOffsetY += mouseY;
					previewOffsetY /= 1.15;
					previewOffsetY -= mouseY;
				}
				
				// Set the transform for the OpenGL canvas just in case we're using it
				DeblurOpenGL.setTransform(previewZoomFactor, -previewOffsetX, previewOffsetY);
				
				redrawPreviewImage();
			}
		});
		
		interactionFrame.addMouseMotionListener(new MouseMotionListener() {
			
			public void mouseDragged(MouseEvent e) {
				
				if (previousMouseX != -1) {
					previewOffsetX += (previousMouseX - e.getX());
					previewOffsetY += (previousMouseY - e.getY());
					
					// Set the transform for the OpenGL canvas just in case we're using it
					DeblurOpenGL.setTransform(previewZoomFactor, -previewOffsetX, previewOffsetY);
					
					redrawPreviewImage();
				}
				
				previousMouseX = e.getX();
				previousMouseY = e.getY();
			}
			
			public void mouseMoved(MouseEvent e) {
				previousMouseX = -1;
				previousMouseY = -1;
			}
		});
		
		interactionFrame.setLayout(new BorderLayout());
		interactionFrame.setCursor(Cursor.getPredefinedCursor(Cursor.CROSSHAIR_CURSOR));
		interactionFrame.add(renderLabel, BorderLayout.CENTER);
		interactionFrame.setBackground(new Color(255, 0, 255));
		frame.add(interactionFrame);
		
		// Create the bottom panel
		final JPanel bottomPanel = new JPanel(new BorderLayout());
		bottomPanel.setPreferredSize(new Dimension(100, 26));
		bottomPanel.setBorder(BorderFactory.createCompoundBorder(
				BorderFactory.createMatteBorder(1, 0, 0, 0, Color.LIGHT_GRAY),
				new EmptyBorder(0, 2, 0, 8)));

		bottomPanel.add(imageInfoLabel, BorderLayout.EAST);
		
		final JPanel leftBottomPanel = new JPanel();
		leftBottomPanel.add(statusLabel);
		progressBar = new JProgressBar(0, 100);
		progressBar.setValue(0);
		progressBar.setStringPainted(false);
		progressBar.setVisible(false);
		
		leftBottomPanel.add(progressBar);
		bottomPanel.add(leftBottomPanel, BorderLayout.WEST);
		frame.add(bottomPanel, BorderLayout.SOUTH);
		
		// Change the way tool tips look
		UIManager.put("ToolTip.background", new Color(255, 255, 255));
		UIManager.put("ToolTip.border", BorderFactory.createLineBorder(new Color(150, 150, 150)));
		
		final int sideWidth = 180;
		
		final JPanel leftPanel = new JPanel();
		leftPanel.setPreferredSize(new Dimension(sideWidth, 10));
		
		final JButton openButton = new JButton("Open Image");
		leftPanel.add(openButton);

		imageNameLabel = new JLabel();
		imageNameLabel.setText("No image selected");
		leftPanel.add(Box.createRigidArea(new Dimension(sideWidth, 0)));
		leftPanel.add(imageNameLabel);
		
		final JLabel adjustmentLabel = new JLabel();
		adjustmentLabel.setText("Image adjustment options:");
		
		final JButton contrastButton = new JButton("Adjust...");
		contrastButton.setEnabled(false);
		contrastButton.setPreferredSize(new Dimension(140, 25));
		contrastButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				
				// Prevent usage with OpenGL
				if (Algorithms.useOpenGL) {
					JOptionPane.showMessageDialog(UserInterface.frame,
							"Image Adjust not implemented in OpenGL",
							"OpenGL Limitation",
							JOptionPane.ERROR_MESSAGE);
					return;
				}
				
				ImageEffects.showEffectDialogue("Adjust", ImageEffects.ADJUST,
						"Contrast", -100, 100, 0, 1,
						"Brightness", -260, 260, 0, 1,
						"Saturation", 0, 2, 1, 10,
						"Exposure adjustment", -100, 100, 0, 1,
						null, 0, 0, 0, 1);
			}
		});
		
		final JButton sharpenButton = new JButton("Sharpen...");
		sharpenButton.setToolTipText("Unsharp mask");
		sharpenButton.setEnabled(false);
		sharpenButton.setPreferredSize(new Dimension(140, 25));
		sharpenButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				
				// Prevent usage with OpenGL
				if (Algorithms.useOpenGL) {
					JOptionPane.showMessageDialog(UserInterface.frame,
							"Sharpen not implemented in OpenGL",
							"OpenGL Limitation",
							JOptionPane.ERROR_MESSAGE);
					return;
				}
				
				ImageEffects.showEffectDialogue("Sharpen", ImageEffects.SHARPEN,
						"Sharpen amount", 0, 10, 1, 10,
						"Radius", 1, 92, 5, 1,
						null, 0, 0, 0, 1,
						null, 0, 0, 0, 1,
						null, 0, 0, 0, 1);
			}
		});
		
		final JButton deblurButton3 = new JButton("Fast Deblur...");
		deblurButton3.setToolTipText("deblurBasic2()");
		deblurButton3.setEnabled(false);
		deblurButton3.setPreferredSize(new Dimension(140, 25));
		deblurButton3.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				ImageEffects.showEffectDialogue("Fast Deblur", ImageEffects.FAST_METHOD,
						"Deblur amount", 0, 5, 1, 10,
						"Radius", 1, 200, 5, 4,
						"Iterations", 1, Algorithms.useOpenGL ? 1 : 10, 1, 1, // OpenGL implementation doesn't support multiple iterations.
						null, 0, 0, 0, 0,
						null, 0, 0, 0, 0);
			}
		});
		
		final JButton deblurButton5 = new JButton("Richardson-Lucy...");
		deblurButton5.setToolTipText("richardsonLucy()");
		deblurButton5.setEnabled(false);
		deblurButton5.setPreferredSize(new Dimension(140, 25));
		deblurButton5.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				
				// Prevent usage with OpenGL
				if (Algorithms.useOpenGL) {
					JOptionPane.showMessageDialog(UserInterface.frame,
							"Richardson-Lucy deconvolution not implemented in OpenGL",
							"OpenGL Limitation",
							JOptionPane.ERROR_MESSAGE);
					return;
				}
				
				ImageEffects.showEffectDialogue("Richardson-Lucy deconvolution", ImageEffects.RICHARDSON_LUCY,
						"Radius", 1, 70, 4, 4,
						"Iterations", 1, 1000, 1, 1,
						null, 0, 0, 0, 0,
						null, 0, 0, 0, 0,
						null, 0, 0, 0, 0);
			}
		});
		
		final JButton deblurButton8 = new JButton("Wiener...");
		deblurButton8.setToolTipText("wienerDeconvolvePublic()");
		deblurButton8.setEnabled(false);
		deblurButton8.setPreferredSize(new Dimension(140, 25));
		deblurButton8.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				
				// Prevent usage with OpenGL
				if (Algorithms.useOpenGL) {
					JOptionPane.showMessageDialog(UserInterface.frame,
							"Wiener deconvolution not implemented in OpenGL",
							"OpenGL Limitation",
							JOptionPane.ERROR_MESSAGE);
					return;
				}
				
				ImageEffects.showEffectDialogue("Wiener deconvolution", ImageEffects.WIENER,
						"Radius", 1, 100, 4, 1,
						"Signal-to-noise ratio", 100, 5100, 2500, 1,
						null, 0, 0, 0, 0,
						null, 0, 0, 0, 0,
						null, 0, 0, 0, 0);
			}
		});
		
		final JButton deblurButton7 = new JButton("Disk Blur...");
		deblurButton7.setToolTipText("Perfect defocus disk blur");
		deblurButton7.setEnabled(false);
		deblurButton7.setPreferredSize(new Dimension(140, 25));
		deblurButton7.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				
				// Prevent usage with OpenGL
				if (Algorithms.useOpenGL) {
					JOptionPane.showMessageDialog(UserInterface.frame,
							"Disk Blur not implemented in OpenGL",
							"OpenGL Limitation",
							JOptionPane.ERROR_MESSAGE);
					return;
				}
				
				ImageEffects.showEffectDialogue("Disk Blur", ImageEffects.DISK_BLUR,
						"Radius", 1, 200, 4, 4,
						null, 0, 0, 0, 0,
						null, 0, 0, 0, 0,
						null, 0, 0, 0, 0,
						null, 0, 0, 0, 0);
			}
		});
		
		final JLabel threadsLabel = new JLabel();
		threadsLabel.setText("CPU threads: ");
		
		// Determine the default number of cores
		Algorithms.numThreads = Math.max(Runtime.getRuntime().availableProcessors(), 1);
		
		final SpinnerModel eyepieceSpinnerModel1 = new SpinnerNumberModel(Algorithms.numThreads, 1, 512, 1);
		final JSpinner threadsSpinner = new JSpinner(eyepieceSpinnerModel1);
		threadsSpinner.setPreferredSize(new Dimension(40, 20));
		threadsSpinner.setOpaque(false);
		threadsSpinner.addChangeListener(new ChangeListener() {
			public void stateChanged(ChangeEvent e) {
				Algorithms.numThreads = (Integer)threadsSpinner.getValue();
			}
		});

		final JRadioButton useGPUCheck = new JRadioButton("GPU rendering");
		final JRadioButton useOpenGLCheck = new JRadioButton("OpenGL rendering");
		final JRadioButton useCPUCheck = new JRadioButton("CPU rendering");
		
		final ActionListener renderRadioButtonListener = new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if (useCPUCheck.isSelected()) {
					Algorithms.useOpenCL = false;
					Algorithms.useOpenGL = false;
				} else {
					Algorithms.useOpenCL = useGPUCheck.isSelected();
					Algorithms.useOpenGL = useOpenGLCheck.isSelected();
				}
				
				if (Algorithms.useOpenGL) {
					enableOpenGLCanvasMode();
				} else {
					disableOpenGLCanvasMode();
				}
			}
		};
		
		useGPUCheck.setSelected(Algorithms.useOpenCL);
		useGPUCheck.addActionListener(renderRadioButtonListener);
		
		useOpenGLCheck.setSelected(Algorithms.useOpenGL);
		useOpenGLCheck.addActionListener(renderRadioButtonListener);
		
		useCPUCheck.setSelected(!Algorithms.useOpenCL && !Algorithms.useOpenGL);
		useCPUCheck.addActionListener(renderRadioButtonListener);
		
		final ButtonGroup buttonGroup = new ButtonGroup();
		buttonGroup.add(useCPUCheck);
		buttonGroup.add(useGPUCheck);
		buttonGroup.add(useOpenGLCheck);
		
		// Create a vertical BoxLayout and add the radio buttons
        final Box renderingModeBox = Box.createVerticalBox();
        renderingModeBox.add(useCPUCheck);
        renderingModeBox.add(useGPUCheck);
        renderingModeBox.add(useOpenGLCheck);
		
		final JCheckBox previewCheckBox = new JCheckBox("Automatically preview");
		previewCheckBox.setSelected(ImageEffects.autoPreviewEnabled);
		previewCheckBox.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				ImageEffects.autoPreviewEnabled = previewCheckBox.isSelected();
			}
		});
		
		final JButton saveButton = new JButton("Save image ...");
		//saveButton.setEnabled(false);
		saveButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				// If a dialog is showing, or something is rendering, then don't allow a save
				if (ImageEffects.isDialogShowing || ImageEffects.isRendering) {
					return;
				}
				
				// Cannot save if using OpenGL for rendering
				if (Algorithms.useOpenGL) {
					displayOpenGLSaveWarning();
					return;
				}
				
				final JFileChooser chooser = new JFileChooser(lastSaveFileDirectory);
				final FileNameExtensionFilter jpgFilter = new FileNameExtensionFilter(
					"JPG Images", "jpg");
				final FileNameExtensionFilter pngFilter = new FileNameExtensionFilter(
					"PNG Images", "png");
				chooser.setFileFilter(jpgFilter);
				chooser.setFileFilter(pngFilter);
				chooser.setMultiSelectionEnabled(false);
				chooser.setDialogTitle("Save image");
				chooser.setPreferredSize(new Dimension(600, 400));
				
				int returnVal = chooser.showSaveDialog(frame);
				
				if (returnVal == JFileChooser.APPROVE_OPTION) {
					File file = chooser.getSelectedFile();
					
					if (file != null) {
						
						// Record the most recent save directory for convenience
						int lastSlashIndex = file.getAbsolutePath().lastIndexOf('\\');
						if (lastSlashIndex != -1) {
							String folder = file.getAbsolutePath().substring(0, lastSlashIndex);
							lastSaveFileDirectory = folder;
						}
						
						final BufferedImage imageToSave = Algorithms.arrayToImage(Algorithms.imageArray);
						
						if (chooser.getFileFilter() == jpgFilter) {
							if (!file.getName().endsWith(".jpg")) {
								file = new File(file.getAbsolutePath() + ".jpg");
							}
							
							// Prompt the user for the compression quality of the JPG image
							float compressionQuality = 0.75f;
							while (true) {
								String ans = JOptionPane.showInputDialog(chooser, "Input JPG quality 0.0 to 1.0 range",
										"JPG Quality", JOptionPane.QUESTION_MESSAGE);
								try {
									float val = Float.parseFloat(ans);
									if (val > 0 && val <= 1) {
										compressionQuality = val;
										break;
									}
								} catch (Exception e){}
							}
							
							JPEGImageWriteParam jpegParams = new JPEGImageWriteParam(null);
							jpegParams.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
							jpegParams.setCompressionQuality(compressionQuality); // Quality range: [0, 1]
							
							final ImageWriter writer = ImageIO.getImageWritersByFormatName("jpg").next();
							
							try {
								FileImageOutputStream fios = new FileImageOutputStream(file);
								
								// specifies where the jpg image has to be written
								writer.setOutput(fios);
								
								// writes the file with given compression level 
								// from your JPEGImageWriteParam instance
								writer.write(null, new IIOImage(imageToSave, null, null), jpegParams);
								
								// Close the output stream
								fios.close();
								
								// Close the writer
								writer.dispose();
							} catch (Exception e) {
								e.printStackTrace();
								
								if (writer != null) {
									writer.dispose();
								}
							}
						} else {
							if (!file.getName().endsWith(".png")) {
								file = new File(file.getAbsolutePath() + ".png");
							}
							
							try {
								ImageIO.write(imageToSave, "png", file);
							} catch (Exception e) {
								e.printStackTrace();
							}
						}
					} else {
						print("No file selected");
					}
				}
			}
		});
		
		final JButton resetImageButton = new JButton("Reload image");
		resetImageButton.setEnabled(false);
		resetImageButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				// If a dialog is showing, or something is rendering, then don't allow this
				if (ImageEffects.isDialogShowing || ImageEffects.isRendering) {
					return;
				}
				
				if (fileDirectory == null) {
					return;
				}
				
				// Reload the image
				loadImage(new File(fileDirectory));
				
				updateProgress(1);
			}
		});
		
		// Add the action listener to the open button
		openButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				// If a dialog is showing, or something is rendering, then don't allow this
				if (ImageEffects.isDialogShowing || ImageEffects.isRendering) {
					return;
				}
				
				final JFileChooser chooser = new JFileChooser(lastFileDirectory);
				FileNameExtensionFilter filter1 = new FileNameExtensionFilter(
					"JPG/PNG/MP4/MPEG", "jpg", "png", "jpeg", "mp4", "m4v", "m4p", "mpg", "mpeg");
				chooser.setFileFilter(filter1);
				chooser.setMultiSelectionEnabled(false);
				chooser.setDragEnabled(true);
				chooser.setDialogTitle("Open image");
				chooser.setPreferredSize(new Dimension(600, 400));
				
				int returnVal = chooser.showOpenDialog(frame);
				
				if (returnVal == JFileChooser.APPROVE_OPTION) {
					final File file = chooser.getSelectedFile();
					
					String fileName = file.getName().toLowerCase();
					
					// Check if this is a video file
					if (fileName.endsWith(".mp4") ||
							fileName.endsWith(".m4v") ||
							fileName.endsWith(".m4p") ||
							fileName.endsWith(".mpg") ||
							fileName.endsWith(".mpeg")) {
						
						// Don't allow video decoding while another is running
            			if (VideoDecoder.isVideoRunning) {
            				return;
            			}
            			
            			final double deblurRadius = promptVideoDeblurRadius();
            			if (deblurRadius <= 0) {
            				return;
            			}
            			
            			// Display the video in a loop
                        VideoDecoder.runLoopedVideoDeconvolution(file.getAbsolutePath(), deblurRadius);
                        
					} else { // Read as image
					
						// Read the image and prepare it
						loadImage(file);
						
						// Enable the manipulation options
	                    resetImageButton.setEnabled(true);
	    				adjustmentLabel.setEnabled(true);
	    				contrastButton.setEnabled(true);
	    				sharpenButton.setEnabled(true);
	    				deblurButton3.setEnabled(true);
	    				deblurButton5.setEnabled(true);
	    				deblurButton7.setEnabled(true);
	    				deblurButton8.setEnabled(true);
	    				saveButton.setEnabled(true);
					}
				}
			}
		});
		
		leftPanel.add(Box.createRigidArea(new Dimension(sideWidth, 1)));
		
		leftPanel.add(resetImageButton);
		
		leftPanel.add(Box.createRigidArea(new Dimension(sideWidth, 1)));
		final JSeparator sep4 = new JSeparator(SwingConstants.HORIZONTAL);
		sep4.setPreferredSize(new Dimension(220, 1));
		leftPanel.add(sep4);
		
		leftPanel.add(adjustmentLabel);
		leftPanel.add(contrastButton);
		leftPanel.add(sharpenButton);
		leftPanel.add(deblurButton3);
		leftPanel.add(deblurButton5);
		leftPanel.add(deblurButton8);
		leftPanel.add(deblurButton7);
		leftPanel.add(threadsLabel);
		leftPanel.add(threadsSpinner);
        leftPanel.add(renderingModeBox);
		leftPanel.add(previewCheckBox);
		
		final JSeparator sep5 = new JSeparator(SwingConstants.HORIZONTAL);
		sep5.setPreferredSize(new Dimension(sideWidth - 20, 5));
		leftPanel.add(sep5);
		
		leftPanel.add(saveButton);
		
		frame.add(leftPanel, BorderLayout.WEST);
		
		// Enable drag and drop of images to the window
		new DropTarget(frame, new DropTargetListener() {
			public void dragEnter(DropTargetDragEvent e) {}
			public void dragExit(DropTargetEvent e) {}
			public void dragOver(DropTargetDragEvent e) {}
			public void dropActionChanged(DropTargetDragEvent dtde) {}
			
			// Called when a file is dragged and dropped onto the "Open" button
			public void drop(DropTargetDropEvent e) {
				// If a dialog is showing, or something is rendering, then don't allow this
				if (ImageEffects.isDialogShowing || ImageEffects.isRendering) {
					return;
				}
				
				try {
	                if (e.isDataFlavorSupported(DataFlavor.javaFileListFlavor)) {
	                    e.acceptDrop(e.getDropAction());

						final Transferable transferable = e.getTransferable();
						@SuppressWarnings("unchecked")
						List<File> files = ((List<File>)transferable.getTransferData(DataFlavor.javaFileListFlavor));
                        if (files != null && files.size() == 1) {
                        	final File file = files.get(0);
                        	
                        	// Check to make sure the file is of a valid type
                        	String name = file.getName().toLowerCase();
                    		if (name.endsWith(".jpg") ||
                    				name.endsWith(".jpeg") ||
                    				name.endsWith(".png") ||
                    				name.endsWith(".bmp")) {
                    			
                    			e.dropComplete(true);
                    			
                    			// Enable the manipulation options
                                resetImageButton.setEnabled(true);
                				adjustmentLabel.setEnabled(true);
                				contrastButton.setEnabled(true);
                				sharpenButton.setEnabled(true);
                				deblurButton3.setEnabled(true);
                				deblurButton5.setEnabled(true);
                				deblurButton7.setEnabled(true);
                				deblurButton8.setEnabled(true);
                				saveButton.setEnabled(true);
            					
        						// Read the image and prepare it
                				loadImage(file);
                				
                    		} else if (name.endsWith(".mp4") ||
                    				name.endsWith(".m4v") ||
                    				name.endsWith(".m4p") ||
                    				name.endsWith(".mpg") ||
                    				name.endsWith(".mpeg")) {
                    			
                    			// Don't allow video decoding while another is running
                    			if (VideoDecoder.isVideoRunning) {
                    				return;
                    			}
                    			
                    			e.dropComplete(true);
                    			
                    			final double deblurRadius = promptVideoDeblurRadius();
                    			if (deblurRadius <= 0) {
                    				return;
                    			}
                    			
                    			// Display the video in a loop
                                VideoDecoder.runLoopedVideoDeconvolution(file.getAbsolutePath(), deblurRadius);
                    		} else {
                    			e.dropComplete(false);
                    		}
                        }
	                } else {
	                    e.rejectDrop();
	                }
				} catch (Exception ex) {
					ex.printStackTrace();
				}
			}
		});
		
		interactionFrame.addComponentListener(new ComponentListener() {
			public void componentResized(ComponentEvent e) {
				if (renderLabel.getWidth() < 1 || renderLabel.getHeight() < 1) {
					return;
				}
				
				X = renderLabel.getWidth();
				Y = renderLabel.getHeight();
				renderImage = new BufferedImage(X, Y, BufferedImage.TYPE_INT_RGB);
				g = renderImage.createGraphics();
				g.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
						RenderingHints.VALUE_ANTIALIAS_ON);
				g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING,
						RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
				
				renderLabel.setIcon(new ImageIcon(renderImage));
				redrawPreviewImage();
			}

			public void componentHidden(ComponentEvent arg0) {}
			public void componentMoved(ComponentEvent arg0) {}
			public void componentShown(ComponentEvent arg0) {}
		});
		
		// This allows the renderLabel to receive key events
		KeyboardFocusManager manager = KeyboardFocusManager.getCurrentKeyboardFocusManager();
		manager.addKeyEventDispatcher(new KeyEventDispatcher() {
			public boolean dispatchKeyEvent(KeyEvent e) {
				if (e.getID() == KeyEvent.KEY_PRESSED && frame.isFocused()) {
					if (e.getKeyCode() == KeyEvent.VK_HOME || e.getKeyCode() == KeyEvent.VK_9
							|| e.getKeyCode() == KeyEvent.VK_ADD) {
						// Zoom the image back to default
						
						// Reset the pan and zoom
						previewOffsetX = 0;
						previewOffsetY = 0;
						previewZoomFactor = Math.min(Math.min(
								(double)(X - 10) / Algorithms.imageArray[0].length,
								(double)(Y - 10) / Algorithms.imageArray[0][0].length), 1);

						// Set the transform for the OpenGL canvas just in case we're using it
						DeblurOpenGL.setTransform(previewZoomFactor, -previewOffsetX, previewOffsetY);
						
						// Redraw the preview of the image
						redrawPreviewImage();
					} else if (e.getKeyCode() == KeyEvent.VK_0 || e.getKeyCode() == KeyEvent.VK_SUBTRACT) {
						// Zoom the image to 100% (pixel perfect)
						double scaleChange = 1/previewZoomFactor;
						previewZoomFactor = 1;
						
						previewOffsetX *= scaleChange;
						previewOffsetY *= scaleChange;

						// Set the transform for the OpenGL canvas just in case we're using it
						DeblurOpenGL.setTransform(previewZoomFactor, -previewOffsetX, previewOffsetY);
						
						// Redraw the preview of the image
						redrawPreviewImage();
						
					} else if (e.getKeyCode() == KeyEvent.VK_V && e.isControlDown()) {
						// Paste an image
						Transferable transferable = Toolkit.getDefaultToolkit().getSystemClipboard().getContents(null);
						if (transferable != null && transferable.isDataFlavorSupported(DataFlavor.imageFlavor)) {
							try {
								BufferedImage temp = (BufferedImage) transferable.getTransferData(DataFlavor.imageFlavor);
								
								// Convert to byte format
								BufferedImage image = new BufferedImage(temp.getWidth(), temp.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
								Graphics2D g = image.createGraphics();
								g.drawImage(temp, 0, 0, null);
								g.dispose();
								
								// Enable the manipulation options
	                            resetImageButton.setEnabled(true);
	            				adjustmentLabel.setEnabled(true);
	            				contrastButton.setEnabled(true);
	            				sharpenButton.setEnabled(true);
	            				deblurButton3.setEnabled(true);
	            				deblurButton5.setEnabled(true);
	            				deblurButton7.setEnabled(true);
	            				deblurButton8.setEnabled(true);
	            				saveButton.setEnabled(true);
	    						
	    						// Read the image and prepare it
	            				fileName = "clipboard";
	            				fileDirectory = null;
	            				
	            				setProcessName("Loaded image " + fileName);
	            				
	            				// Show the image filename
	            				imageNameLabel.setText(fileName);
	            				
	            				// Load the image
	            				loadBufferedImage(image);
	            				
							} catch (Exception e2) {
								e2.printStackTrace();
							}
						}
					}
				}
				
				return false;
			}
		});
		
		// If we are going to use OpenGL to render the images, then swap out the normal canvas for the GLCanvas
		if (Algorithms.useOpenGL) {
			enableOpenGLCanvasMode();
		}
		
		frame.setMinimumSize(new Dimension(400, 200));
		frame.setLocation(30, 30);
		frame.pack();
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
	}
	
	// Easy print function
	private static void print(final Object o) {
		System.out.println(o);
	}
}



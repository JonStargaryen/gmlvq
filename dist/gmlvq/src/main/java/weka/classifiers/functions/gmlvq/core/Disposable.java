package weka.classifiers.functions.gmlvq.core;

import java.io.Closeable;
import java.util.concurrent.ExecutorService;

/**
 * Provides a signature which enables wrapping classes to safely dispose
 * resources internally used by the implementing class - somewhat like
 * {@link Closeable} does for opened files.<br />
 * In GMLVQ implementation, it is assigned to classes which utilize
 * {@link ExecutorService} so the internal thread pools can be shutdown safely.
 * <br />
 *
 * @author S
 *
 */
public interface Disposable {

    /**
     * frees resources associated to this object
     * 
     * @throws InterruptedException
     */
    void dispose() throws InterruptedException;
}

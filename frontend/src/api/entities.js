import { annotationAPI, Annotation as AnnotationInterface } from './awsClient';

// Export the Annotation interface for type checking
export { AnnotationInterface as Annotation };

// Export the annotation API as AnnotationEntity for backward compatibility
export const AnnotationEntity = annotationAPI;
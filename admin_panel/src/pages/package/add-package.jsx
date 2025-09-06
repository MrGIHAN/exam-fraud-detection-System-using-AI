import React from 'react';
import PageHeader from '@/components/shared/pageHeader/PageHeader';
import Footer from '@/components/shared/Footer';
import AddPackageForm from '@/components/package/AddPackageForm';

const AddPackage = () => {
    return (
        <>
            <PageHeader >
            </PageHeader>
            <div className='main-content'>
                <div className='row'>
                    <AddPackageForm title={"Add a new student"} />
                </div>
            </div>
            <Footer />
        </>
    );
}

export default AddPackage;